# 用于data（LandSat8）

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from datasets.own_data import ImageFolder
from util import test_single_volume
from torchsummary import summary

#from networks.CDCTFM import CDCTFM
#from networks.CDCTFM_without_ATT import CDCTFM
#from networks.CDCTFM_without_dark import CDCTFM
#from networks.CDCTFM_mobile import CDCTFM
#from networks.CDCTFM_ASPP import CDCTFM
#from networks.CDCTFM_SE import CDCTFM

from networks.非轻量级.CRSNet import CRSNet



import cv2
import re
from numpy import *

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data/val', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=200,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", default=True, help='whether to save results during inference')
parser.add_argument('--base_size', type=int,
                    default=384, help='input patch size of original input')
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()



def inference(args, model, test_save_path=None):
    db_test = args.Dataset(args, split="validation")
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    print("{} test iterations per epoch".format(len(testloader)))
    model.eval()
 
 
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, case_name = sampled_batch['image'], sampled_batch['case_name'][0]

        image = image.squeeze(0).cpu().detach().numpy()

        with torch.no_grad():
            input = torch.from_numpy(image).unsqueeze(0).float().cuda()
            outputs = net(input)
            out = torch.argmax(torch.sigmoid(outputs), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()*255
        cv2.imwrite(test_save_path + '/'+ case_name + '.jpg', out)
    
   

    return "Testing Finished!"

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
        'own': {
            'Dataset': ImageFolder,
            'root_path': '../data/',
            'volume_path': './data/',
            'list_dir': '',
            'num_classes': 2,
            'z_spacing': 1,
        },
    }

    dataset_name = 'own'
    args.root_path = dataset_config[dataset_name]['root_path']
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.Dataset = dataset_config[dataset_name]['Dataset']


    #net = CDCTFM(in_channel=4).cuda()
    net = CRSNet(in_channel=4).cuda()
    net.load_state_dict(torch.load("/hy-tmp/prj/model/TU_own224/TU_epo150_bs32_lr0.01_CRSNet_final/epoch_80.pth"))
    
    args.test_save_dir = './predictions'
    test_save_path = os.path.join(args.test_save_dir, "CRSNet70")
    os.makedirs(test_save_path, exist_ok=True)
    
    inference(args, net, test_save_path)
