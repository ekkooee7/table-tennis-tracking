import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from numba import jit

import COLOR_library
from DIP_project import IMAGE_PROCESS

if __name__ == '__main__':
    img = cv2.imread('img_3.png')

    process_img = IMAGE_PROCESS.HSV_process(img)

    cv2.imshow('original image', img)
    # cv2.imshow('1', img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow('hsv_result', process_img)

    row, col = img.shape[0], img.shape[1]
    print('img size is')
    print(row, col)

    H = np.zeros([row, col])
    S = np.zeros([row, col])
    V = np.zeros([row, col])
    t0 = time.time()
    [H, S, V] = COLOR_library.RGB_HSV_jit(img, H, S, V)
    t1 = time.time()
    print('processing time 1st is:' + str(round(t1 - t0, 3)))


    ##############################################################
    ## 获得与cv一样的hsv image， 方便调试
    ##############################################################
    cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv_cv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img_hsv = np.zeros([row, col, 3], 'uint8')
    img_hsv[:, :, 0] = H
    img_hsv[:, :, 1] = S
    img_hsv[:, :, 2] = V

    # lower = np.array([0, 144, 184])
    # upper = np.array([28, 255, 255])
    #
    # # 返回黄色区域的二值图像
    # img_range = cv2.inRange(img_hsv, lower, upper)

    cv2.imshow("hsv", img_hsv)
    cv2.imshow("hsv cv", img_hsv_cv)
    ##############################################################

    R = np.zeros([row, col])
    G = np.zeros([row, col])
    B = np.zeros([row, col])

    np.clip(H, 0, 360, out=H)
    np.clip(S, 0, 100, out=S)
    np.clip(V, 0, 100, out=V)

    t2 = time.time()
    [R, G, B] = COLOR_library.HSV_RGB_jit(H, S, V, R, G, B)
    R = np.multiply(R, 255)
    G = np.multiply(G, 255)
    B = np.multiply(B, 255)
    t3 = time.time()
    print('processing time 2th is:' + str(round(t3 - t2, 3)))

    np.clip(H, 0, 255, out=H)
    np.clip(S, 0, 255, out=S)
    np.clip(V, 0, 255, out=V)

    np.clip(R, 0, 255, out=R)
    np.clip(G, 0, 255, out=G)
    np.clip(B, 0, 255, out=B)

    rec_img = np.zeros([row, col, 3], dtype='uint8')
    rec_img[:, :, 0] = B
    rec_img[:, :, 1] = G
    rec_img[:, :, 2] = R

    # cv2.imshow("H", H)
    # cv2.imshow("S", S)
    # cv2.imshow("V", V)

    # cv2.imshow("recover image", rec_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
