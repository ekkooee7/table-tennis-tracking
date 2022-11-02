import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from numba import jit

import COLOR_library
from DIP_project import IMAGE_PROCESS

@jit(nopython=True)
def RGB_HSV_jit(img, H, S, V):
    '''
    accerelate using the numba.jit
    根据cv文档描述编写的HSV算法
    如果没有安装numba.jit包可以注释掉 @jit(nopython=True)

    H: 0-180
    S: 0-255
    V: 0-255
    '''
    row, col = img.shape[0], img.shape[1]

    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    for i in range(row):
        for j in range(col):
            R = int(img_r[i, j]) / 255
            G = int(img_g[i, j]) / 255
            B = int(img_b[i, j]) / 255

            C_max = max(R, G, B)
            C_min = min(R, G, B)
            delta = C_max - C_min

            if C_max == 0:
                S[i, j] = 0
            elif C_max != 0:
                S[i, j] = delta / C_max * 255

            V[i, j] = C_max * 255

            if delta == 0:
                H[i, j] = 0
            elif C_max == R:
                H[i, j] = 60 * ((G - B) / delta)
            elif C_max == G:
                H[i, j] = 60 * ((B - R) / delta + 2)
            elif C_max == B:
                H[i, j] = 60 * ((R - G) / delta + 4)

            if H[i, j] < 0:
                H[i, j] = H[i, j] + 360

            H[i, j] = H[i, j] * 0.5

    return [H, S, V]


def RGB_HSV_ori(img, H, S, V):
    '''
    accerelate using the numba.jit
    根据cv文档描述编写的HSV算法
    如果没有安装numba.jit包可以注释掉 @jit(nopython=True)

    H: 0-180
    S: 0-255
    V: 0-255
    '''
    row, col = img.shape[0], img.shape[1]

    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    for i in range(row):
        for j in range(col):
            R = int(img_r[i, j]) / 255
            G = int(img_g[i, j]) / 255
            B = int(img_b[i, j]) / 255

            C_max = max(R, G, B)
            C_min = min(R, G, B)
            delta = C_max - C_min

            if C_max == 0:
                S[i, j] = 0
            elif C_max != 0:
                S[i, j] = delta / C_max * 255

            V[i, j] = C_max * 255

            if delta == 0:
                H[i, j] = 0
            elif C_max == R:
                H[i, j] = 60 * ((G - B) / delta)
            elif C_max == G:
                H[i, j] = 60 * ((B - R) / delta + 2)
            elif C_max == B:
                H[i, j] = 60 * ((R - G) / delta + 4)

            if H[i, j] < 0:
                H[i, j] = H[i, j] + 360

            H[i, j] = H[i, j] * 0.5

    return [H, S, V]

if __name__ == '__main__':
    IMG_PATH = 'img_3.png'
    img1 = cv2.imread(IMG_PATH)
    # img2 = cv2.imread(IMG_PATH)
    # img3 = cv2.imread(IMG_PATH)

    dim = (4000, 4000)
    # resize image
    img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    img3 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)

    row, col = img1.shape[0], img1.shape[1]

    H = np.zeros([row, col])
    S = np.zeros([row, col])
    V = np.zeros([row, col])

    t0 = time.time()
    process_img0 = RGB_HSV_jit(img1, H, S, V)
    t1 = time.time()

    process_img1 = RGB_HSV_ori(img2, H, S, V)
    t2 = time.time()

    process_img2 = cv2.cvtColor(img3, cv2.COLOR_RGB2HSV)
    t3 = time.time()

    print(t1 - t0)
    print(t2 - t1)
    print(t3 - t2)

    # cv2.imshow("0", process_img0)


    cv2.waitKey(0)
    cv2.destroyAllWindows()