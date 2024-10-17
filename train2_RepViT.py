import os
import sys
import argparse
import logging
import random
import numpy as np

import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import DiceLoss

from datasets.own_data import ImageFolder

from network.repvit import *

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=2, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
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
            'root_path': '../data/',
            'num_classes': 2,
        },
        'own2': {
            'root_path': '../data2/',
            'num_classes': 2,
        },
    }
    args.root_path = dataset_config[dataset_name]['root_path']

    # 设置保存路径
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_' + 'RepViT'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # 模型
    model = repvit_m1_1(img_size=args.img_size, num_classes=2, in_channel=10).cuda()
    # model.load_state_dict(torch.load("/hy-tmp/prj/model/TU_own224/TU_epo100_bs32_lr0.01_CRSNet/epoch_100.pth"))

    # model = CRSNet(img_size=args.img_size, num_classes=2, in_channel=4).cuda()
    # model.load_state_dict(torch.load("/hy-tmp/prj/model/TU_own2224/TU_epo50_bs32_lr0.01_CRSNet/epoch_50.pth"))

    # 日志
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # 初始学习率
    base_lr = args.base_lr
    # 类别数
    num_classes = dataset_config[dataset_name]['num_classes']
    # batch大小
    batch_size = args.batch_size

    db_train = ImageFolder(args, split="train")
    db_validation = ImageFolder(args, split="validation")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn, drop_last=True)
    ###
    validationloader = DataLoader(db_validation, batch_size=1, num_workers=0, pin_memory=True,
                                  worker_init_fn=worker_init_fn)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001)
    # optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.99), weight_decay=0.001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    ans = ""

    # 开始训练
    for epoch_num in iterator:
        epoch_num = epoch_num + 1

        loss_sum = 0
        i = 0
        model.train()

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            image_batch, label_batch = image_batch.to(torch.device("cuda:0")), label_batch.to(torch.device("cuda:0"))
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=False)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            loss_sum = loss_sum + loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                lr_ = param_group['lr']

            iter_num = iter_num + 1
            i = i + 1
            logging.info('iteration %d : loss : %f, loss_ce: %f, lr: %f' % (iter_num, loss.item(), loss_ce.item(), lr_))
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

    logging.info(ans)

    iterator.close()
