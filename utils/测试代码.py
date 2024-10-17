import cv2
import numpy as np

def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))  # 使用opencv的erode函数更高效

if __name__ == '__main__':
    m = cv2.imread("C:\\Users\\10037\\Desktop\\3.png")
    cv2.imshow("src", m)
    print(m.shape)
    r = 3
    V1 = np.min(m, 2)  # 得到暗通道图像
    cv2.imshow("dst", zmMinFilterGray(V1, r))
    cv2.waitKey(-1)
    cv2.destroyAllWindows()