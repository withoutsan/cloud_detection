import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from network.DATransUNet.DATransUNet import DA_Transformer
from network.DATransUNet.DATransUNet import CONFIGS as CONFIGS_ViT_seg
import sys
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import DiceLoss
from datasets.own_data2 import ImageFolder

from torchvision import transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()

# parser.add_argument('--list_dir', type=str,
#                     default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')


parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--base_size', type=int,
                    default=384, help='input patch size of original input')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=3407, help='random seed')
args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

def trainer_synapse(args, model, snapshot_path):
    # 日志
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # 初始学习率
    base_lr = args.base_lr
    # 类别数
    num_classes = args.num_classes
    # batch 大小
    batch_size = args.batch_size * args.n_gpu

    # 数据集
    db_train = ImageFolder(args, split="train")
    db_validation = ImageFolder(args, split="validation")
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of validation set is: {}".format(len(db_validation)))
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    ###
    validationloader = DataLoader(db_validation, batch_size=1, num_workers=0, pin_memory=True,
                                  worker_init_fn=worker_init_fn)

    # 设置损失函数
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    # 设置优化器
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    # 进度条
    iterator = tqdm(range(max_epoch), ncols=70)
    ans = ""

    writer = SummaryWriter(snapshot_path + '/log')


    for epoch_num in iterator:
        epoch_num = epoch_num + 1

        loss_sum = 0
        i = 0
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # 将数据从CPU发送到GPU
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            # print(image_batch)
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            i = i + 1
            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        if dataset_name == 'own2':
            model.eval()
            for i_batch, sampled_batch in enumerate(validationloader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

                image_batch, label_batch = image_batch.to(torch.device("cuda:0")), label_batch.to(
                    torch.device("cuda:0"))
                outputs = model(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=False)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                loss_sum = loss_sum + loss.item()
                i = i + 1

        ans = ans + str(epoch_num) + ": " + str(loss_sum / i) + "\n"
        logging.info(ans)

        if epoch_num % 10 == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # if epoch_num >= max_epoch - 1:
        #     save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    # cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法
    torch.backends.cudnn.benchmark = True
    # 设置随机数，并使结果是确定的
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 数据集配置
    dataset_name = 'own2'
    dataset_config = {
        'own': {
            'root_path': './data/',
            'num_classes': 2,
        },
        'own2': {
            'root_path': './data2/',
            'num_classes': 2,
        },
    }
    args.root_path = dataset_config[dataset_name]['root_path']

    #     if args.batch_size != 24 and args.batch_size % 6 == 0:
#         args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    # args.list_dir = dataset_config[dataset_name]['list_dir']

    args.exp = 'TU_' + dataset_name + '_' + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    # args.is_pretrain = True
    # snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    # snapshot_path += '_' + args.vit_name
    # snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    # snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    # snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    # snapshot_path = snapshot_path + '_'+str(args.img_size)
    # snapshot_path = snapshot_path + '_'+str(args.seed) if args.seed!=1234 else snapshot_path
    snapshot_path = snapshot_path + '_' + 'DATransUNet'

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = DA_Transformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # pretrained_path = config_vit.pretrained_path
    # # 加载预训练权重
    # weights = np.load(pretrained_path)
    # # 打印预训练权重文件的内容
    # print(weights.files)

    # # net.load_from(weights=np.load(config_vit.pretrained_path))
    # net.load_from(weights=weights)

    trainer = {'own2': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)
