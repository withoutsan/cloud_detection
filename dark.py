# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

import numpy as np
import cv2
import copy

r = 60


def original(i, j, k, ksize, img):
    # 找到矩阵坐标
    x1 = y1 = -ksize // 2
    x2 = y2 = ksize + x1
    temp = np.zeros(ksize * ksize)
    count = 0
    # 处理图像
    for m in range(x1, x2):
        for n in range(y1, y2):
            if i + m < 0 or i + m > img.shape[0] - 1 or j + n < 0 or j + n > img.shape[1] - 1:
                temp[count] = img[i, j, k]
            else:
                temp[count] = img[i + m, j + n, k]
            count += 1
    return temp


def max_min_functin(ksize, img, flag):
    img0 = copy.copy(img)
    for i in range(0, img.shape[0]):
        print(i)
        for j in range(2, img.shape[1]):
            for k in range(img.shape[2]):
                temp = original(i, j, k, ksize, img0)
                img[i, j, k] = np.min(temp)
    return img


x1 = cv2.imread("C:\\Users\\10037\\Desktop\\111\\1.jpg")
x2 = np.min(x1, axis=2).reshape((x1.shape[0], x1.shape[1], 1))
x3 = cv2.erode(x2, np.ones((2*r+1, 2*r+1)))

cv2.imwrite("C:\\Users\\10037\\Desktop\\111\\2.jpg", x2)
cv2.imwrite("C:\\Users\\10037\\Desktop\\111\\3.jpg", x3)
cv2.imshow("a", x3)
cv2.waitKey(0)
