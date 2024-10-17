# 用于data2（MODIS）

import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.own_data2 import ImageFolder

from network.CDCTFM import CDCTFM
# from networks.CDCTFM_without_dark import CDCTFM
# from networks.CDCTFM_mobile import CDCTFM
# from networks.CDCTFM_ASPP import CDCTFM
# from networks.消融实验.注意力.CDCTFM_without_ATT import CDCTFM

# from networks.CRSNet import CRSNet


import os
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data3', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data3/val', help='root dir for validation volume data')  # for acdc volume_path=root_dir

parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int, default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=200,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", default=True, help='whether to save results during inference')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=3407, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def mean(a):
    return sum(a) / len(a)


def inference(args, model, name=None):
    db_test = args.Dataset(args, split="test")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    print("{} tests".format(len(testloader)))
    model.eval()

    # evaluate
    Jaccard_ = []
    Precision_ = []
    Recall_ = []
    F1_ = []
    OA_ = []
    Specificity_ = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]

        image = image.squeeze(0).cpu().detach().numpy()
        actual = label.squeeze(0).squeeze(0).cpu().detach().numpy()

        with torch.no_grad():
            input = torch.from_numpy(image).unsqueeze(0).float().cuda()
            outputs = net(input)
            out = torch.argmax(torch.sigmoid(outputs), dim=1).squeeze(0)
            predict = out.cpu().detach().numpy()
            Jaccard = []
            Precision = []
            Recall = []
            F1 = []
            OA = []
            flag = False
            for c in range(args.num_classes):
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                for a, p in zip(actual.flatten(), predict.flatten()):
                    if a == c and p == c:
                        TP = TP + 1
                    elif a == c and p != c:
                        FN = FN + 1
                    elif a != c and p != c:
                        TN = TN + 1
                    else:
                        FP = FP + 1
                if TP + FP + FN == 0 or TP + FP == 0 or TP + FN == 0 or TN + FP == 0 or TP == 0:
                    flag = True
                    break
                Jaccard.append(TP / (TP + FP + FN))
                Precision.append(TP / (TP + FP))
                Recall.append(TP / (TP + FN))
                p = (TP / (TP + FP))
                r = (TP / (TP + FN))
                F1.append(2 * p * r / (p + r))
                OA.append((TP + TN) / (TP + FP + TN + FN))
            if flag:
                continue
            Jaccard_.append(mean(Jaccard))
            Precision_.append(mean(Precision))
            Recall_.append(mean(Recall))
            F1_.append(mean(F1))
            OA_.append(mean(OA))

    print('Jaccard: %f\n' % (mean(Jaccard_)))
    print('Precesion: %f\n' % (mean(Precision_)))
    print('Recall: %f\n' % (mean(Recall_)))
    print('F1: %f\n' % (mean(F1_)))
    print('OA: %f\n' % (mean(OA_)))

    # 拼图的部分了
    #1. get keys and gts
    gts = {}
    preds = {}
    for i, file_name in enumerate(os.listdir("../gt/")):
        path = os.path.join("../gt/", file_name)
        gt = cv2.imread(path, -1)
        scid = file_name[:-4]
        gts[scid] = gt 
        preds[scid] = np.zeros((512*5, 512*3))
        
    #2. predict and cat
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, case_name = sampled_batch['image'], sampled_batch['case_name'][0]
        scid = case_name[:case_name.find("_")]
        pos = int(case_name[case_name.find("_")+1:])-1
        row = pos//3
        col = pos - 3*row

        image = image.squeeze(0).cpu().detach().numpy()
        with torch.no_grad():
            input = torch.from_numpy(image).unsqueeze(0).float().cuda()
            outputs = net(input)
            out = torch.argmax(torch.sigmoid(outputs), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()*255
            out = cv2.resize(out, (512, 512), interpolation=cv2.INTER_LINEAR_EXACT)

            for i in range(512):
                preds[scid][row*512+i][col*512:(col+1)*512] = out[i]
    
    # 3. crop and save(gt:2030*1354, pred:2560*1536)
    # 获取预测结果
    for scid in gts.keys():
        h, w = gts[scid].shape

        img = np.zeros((h, w))
        for i in range(h):
            img[i] = preds[scid][(512*5-h)//2+i][(512*3-w)//2:(512*3-w)//2+w]

        dirs = "/MODIS/" + name + "/"
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        cv2.imwrite(dirs + scid + '.jpg', img)

    # return "Testing Finished!"


if __name__ == "__main__":
    # cuDNN来衡量自己库里面的多个卷积算法的速度，然后选择其中最快的那个卷积算法
    torch.backends.cudnn.benchmark = True
    # 设置随机数，并使结果是确定的
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    dataset_config = {
        'own2': {
            'Dataset': ImageFolder,
            'root_path': '../data3/',
            'volume_path': '../data3/',
            'list_dir': '',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }

    dataset_name = 'own2'
    args.root_path = dataset_config[dataset_name]['root_path']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.Dataset = dataset_config[dataset_name]['Dataset']

    net = CDCTFM(img_size=args.img_size, num_classes=2, in_channel=10).cuda()
    # net.load_state_dict(torch.load("/hy-tmp/prj/model/TU_own2224/TU_epo50_bs16_lr0.01_CDCTFM/CDCTFM.pth"))
    # net = CRSNet(img_size=args.img_size, num_classes=2, in_channel = 10).cuda()
    net.load_state_dict(
        torch.load("./model/TU_own2224/TU_epo150_bs32_lr0.01_CDCTFM/epoch_20.pth"))
    inference(args, net)
