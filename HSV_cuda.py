import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from numba import jit



def RGB_HSV_cuda(img):
    '''cuda version'''
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    row, col = img.shape[0], img.shape[1]
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    H = cp.zeros([row, col], dtype='uint8')
    S = cp.zeros([row, col], dtype='uint8')
    V = cp.zeros([row, col], dtype='uint8')

    # img_b, img_g, img_r = cv2.split(img)

    for i in range(row):
        for j in range(col):
            R = int(img_r[i, j]) / 255
            G = int(img_g[i, j]) / 255
            B = int(img_b[i, j]) / 255
            C_max = max(R, G, B)
            C_min = min(R, G, B)
            delta = C_max - C_min

            # print(C_max)

            if delta == 0:
                H0 = 0
            elif C_max == R:
                H0 = 60 * ((G - B) / delta)
            elif C_max == G:
                H0 = 60 * ((B - R) / delta + 2)
            elif C_max == B:
                H0 = 60 * ((R - G) / delta + 4)

            if H0 < 0:
                H[i, j] = H0 + 360
            else:
                H[i, j] = H0

            if C_max == 0:
                S[i, j] = 0
            elif C_max != 0:
                S[i, j] = delta / C_max

            V[i, j] = C_max

    return [H, S, V]



if __name__ == '__main__':
    img = cv2.imread('img.png')
    # cv2.imshow('1', img)
    row, col = img.shape[0], img.shape[1]
    print('img size is')
    print(row, col)

    t0 = time.time()
    H, S, V = RGB_HSV_cuda(img)
    t1 = time.time()
    print('processing time is:' + str(round(t1-t0,3)))

    cp.clip(H, 0, 255, out=H)
    cp.clip(S, 0, 255, out=S)
    cp.clip(V, 0, 255, out=V)

    H_np = cp.asnumpy(H)
    S_np = cp.asnumpy(S)
    V_np = cp.asnumpy(V)

    cv2.imshow("H", H_np)
    cv2.imshow("S", S_np)
    cv2.imshow("V", V_np)

    cv2.waitKey()
    cv2.destroyAllWindows()





