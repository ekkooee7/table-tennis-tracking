import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from numba import jit
from collections import defaultdict


@jit(nopython=True)
def RGB_HSV_jit(img, H, S, V):
    '''
    accerelate using the numba.jit
    根据cv文档描述编写的HSV算法
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


def HSV_RGB_V0(H, S, V, R, G, B):
    '''HSV转化到RGB的算法:
    if (s = 0)
    R=G=B=V;
    else
    H /= 60;
    i = INTEGER(H);
    f = H - i;
    a = V * ( 1 - s );
    b = V * ( 1 - s * f );
    c = V * ( 1 - s * (1 - f ) );
    switch(i)
    case 0: R = V; G = c; B = a;
    case 1: R = b; G = v; B = a;
    case 2: R = a; G = v; B = c;
    case 3: R = a; G = b; B = v;
    case 4: R = c; G = a; B = v;
    case 5: R = v; G = a; B = b;
    '''
    row, col = H.shape[0], H.shape[1]

    # np.clip(H, 0, 255, out=H)
    # np.clip(S, 0, 255, out=S)
    # np.clip(V, 0, 255, out=V)

    for i in range(row):
        for j in range(col):
            h, s, v = H[i, j], S[i, j], V[i, j]
            if s == 0:
                r, g, b = v, v, v
            else:
                h = h / 60
                i = int(h)
                f = h - i
                x = v * (1 - s)
                y = v * (1 - s * f)
                z = v * (1 - s * (1 - f))

                if i == 0:
                    r, g, b = v, z, x
                elif i == 1:
                    r, g, b = y, v, x
                elif i == 2:
                    r, g, b = x, v, z
                elif i == 3:
                    r, g, b = x, y, v
                elif i == 4:
                    r, g, b = z, x, v
                elif i == 5:
                    r, g, b = v, x, y

            R[i, j], G[i, j], B[i, j] = r, g, b

    return R, G, B


@jit(nopython=True)
def HSV_RGB_jit(H, S, V, R, G, B):
    '''
    HSV转化到RGB的算法: version 2
    '''
    row, col = H.shape[0], H.shape[1]

    for i in range(row):
        for j in range(col):
            h, s, v = H[i, j], S[i, j], V[i, j]
            c = v * s
            x = c * (1 - abs(np.mod(h / 60, 2) - 1))
            m = v - c
            if 0 <= h < 60:
                r, g, b = c, x, 0
            elif 60 <= h < 120:
                r, g, b = x, c, 0
            elif 120 <= h < 180:
                r, g, b = 0, c, x
            elif 180 <= h < 240:
                r, g, b = 0, x, c
            elif 240 <= h < 300:
                r, g, b = x, 0, c
            elif 300 <= h <= 360:
                r, g, b = c, 0, x

            R[i, j], G[i, j], B[i, j] = r + m, g + m, b + m

    return R, G, B


def get_cord(frame):
    row, col = frame.shape[0], frame.shape[1]
    img_mask = HSV_process(frame)  ## hsv method of mine
    # cv2.imshow('hsv img', img_mask)

    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    x, y, w, h = -1, -1, -1, -1
    targetPos_x, targetPos_y = -1, -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if 20 < area < 300:
            x, y, w, h = cv2.boundingRect(cnt)
            x_total = 0
            y_total = 0

            for i in range(len(cnt[0])):
                x_total = cnt[0][0][0] + x_total
                y_total = cnt[0][0][1] + y_total

            targetPos_x = int(x + w / 2)
            targetPos_y = int(y + h / 2)

            cv2.putText(frame, "({:0<2d}, {:0<2d})".format(targetPos_x, targetPos_y), (20, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)  # 文字
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # locol_img = np.zeros([100, 100, 3])

    if targetPos_x == -1 or targetPos_y == -1:
        pass
        # print('not found')
    elif (targetPos_x - 25 > 0 and targetPos_x + 25 < row - 1
          and targetPos_y - 25 > 0 and targetPos_x + 25 < col - 1):
        local_img = frame[targetPos_y - 25:targetPos_y + 25, targetPos_x - 25:targetPos_x + 25, :]
        cv2.imshow('local image', local_img)

        output_img = local_img.copy()

        circles = hough_detect_ball(input_img=local_img, r_min=8,
                                    r_max=15, delta_r=1, num_thetas=60,
                                    bin_threshold=0.4, min_edge_threshold=50,
                                    max_edge_threshold=100)

        for x, y, r, v in circles:
            output_img = cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)

        cord_x, cord_y = circles[0][0] + targetPos_x - 25, circles[0][1] + targetPos_y - 25

        # print('best observation')
        # print(cord_x, cord_y)
        return cord_x, cord_y

    return [-1, -1]


def hough_detect_ball(input_img, r_min=8, r_max=20, delta_r=1, num_thetas=30,
                      bin_threshold=0.4, min_edge_threshold=50, max_edge_threshold=100):
    # edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    edge_image = cv2.Canny(input_img, min_edge_threshold, max_edge_threshold)

    if edge_image is not None:
        circles = find_hough_circles_fast(input_img, edge_image, r_min, r_max, delta_r, num_thetas,
                                          bin_threshold)

    return circles


def find_hough_circles_fast(image, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process=True):
    # image size
    img_height, img_width = edge_image.shape[:2]

    # R and Theta ranges
    dtheta = int(360 / num_thetas)

    ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
    thetas = np.arange(0, 360, step=dtheta)

    ## Radius ranges from r_min to r_max
    rs = np.arange(r_min, r_max, step=delta_r)

    # Calculate Cos(theta) and Sin(theta) it will be required later
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    # Evaluate and keep ready the candidate circles dx and dy for different delta radius
    # based on the the parametric equation of circle.
    # x = x_center + r * cos(t) and y = y_center + r * sin(t),
    # where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)
    circle_candidates = []
    for r in rs:
        for t in range(num_thetas):
            # instead of using pre-calculated cos and sin theta values you can calculate here itself by following
            # circle_candidates.append((r, int(r*cos(2*pi*t/num_thetas)), int(r*sin(2*pi*t/num_thetas))))
            # but its better to pre-calculate and use it here.
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))

    # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not
    # aready present in the dictionary instead of throwing exception.
    accumulator = defaultdict(int)

    # accumulator = vote(edge_image, accumulator, circle_candidates, img_height, img_width)

    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0:  # white pixel
                # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    accumulator[(x_center, y_center, r)] += 1  # vote for current candidate

    # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold)
    out_circles = []
    # print()
    # print()
    # print("===========================")

    # Sort the accumulator based on the votes for the candidate circles
    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold:
            # Shortlist the circle for final result
            out_circles.append((x, y, r, current_vote_percentage))
            # print(x, y, r, current_vote_percentage)

    # print("===========================")
    # print()
    # print()

    # Post process the results, can add more post processing later.
    if post_process:
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in out_circles:
            # Exclude circles that are too close of each other
            # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
            # Remove nearby duplicate circles based on pixel_threshold
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for
                   xc, yc, rc, v in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
        out_circles = postprocess_circles

    return out_circles


def HSV_process(img):
    '''用于处理输入img， 输出给定的HSV范围内的RGB图像'''
    # img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    row, col = img.shape[0], img.shape[1]

    H = np.zeros([row, col])
    S = np.zeros([row, col])
    V = np.zeros([row, col])
    # R = np.zeros([row, col])
    # G = np.zeros([row, col])
    # B = np.zeros([row, col])

    H, S, V = RGB_HSV_jit(img, H, S, V)

    # H = np.multiply(H, 0.5)
    # S = np.multiply(S, 2.55)
    # V = np.multiply(V, 2.55)

    # np.clip(H, 0, 180, out=H)
    # np.clip(S, 0, 255, out=S)
    # np.clip(V, 0, 255, out=V)

    img_hsv = np.zeros([row, col, 3])

    img_hsv[:, :, 0] = H
    img_hsv[:, :, 1] = S
    img_hsv[:, :, 2] = V

    lower = np.array([0, 184, 202])
    upper = np.array([66, 255, 255])

    # 返回黄色区域的二值图像
    img_hsv = cv2.inRange(img_hsv, lower, upper)

    # H[np.where(H > 27)] = 0
    # S[np.where(S < 140)] = 0
    # V[np.where(V > 30)] = 0
    # H[np.where(0 < H < 27)] = 1
    # S[np.where(141 < S < 255)] = 1
    # V[np.where(30 < V < 255)] = 1

    # [R, G, B] = HSV_RGB_jit(H, S, V, R, G, B)
    # R = np.multiply(R, 255)
    # G = np.multiply(G, 255)
    # B = np.multiply(B, 255)
    #
    # np.clip(R, 0, 255, out=R)
    # np.clip(G, 0, 255, out=G)
    # np.clip(B, 0, 255, out=B)
    #
    # rec_img = np.zeros([row, col, 3], dtype='uint8')
    # rec_img[:, :, 0] = R
    # rec_img[:, :, 1] = G
    # rec_img[:, :, 2] = B

    return img_hsv


def get_hsv_img_cv(img):
    cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 144, 184])
    upper = np.array([28, 255, 255])

    # 返回黄色区域的二值图像
    img_range = cv2.inRange(img_hsv, lower, upper)

    # H = img[:, :, 0]
    # H[np.where(H < 27)] = 0
    # H[np.where(H >= 27)] = 1
    # np.clip(H, 0, 255, out=H)
    return img_range


def hough_circle_detect(img):
    """
    霍夫圆检测
    cv2.HoughCircles()，函数返回值为圆心坐标（x,y）圆半径R。
    其函数原型为： HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    重点参数解析：
        method：定义检测图像中圆的方法。目前唯一实现的方法是cv2.HOUGH_GRADIENT；
        dp：累加器分辨率与图像分辨率的反比。dp获取越大，累加器数组越小；
        minDist：检测到的圆的中心，（x,y）坐标之间的最小距离。如果minDist太小，则可能导致检测到多个相邻的圆。如果minDist太大，
        则可能导致很多圆检测不到；
        param1：用于处理边缘检测的梯度值方法；
        param2：cv2.HOUGH_GRADIENT方法的累加器阈值。阈值越小，检测到的圈子越多；
        minRadius：半径的最小大小（以像素为单位）；
        maxRadius：半径的最大大小（以像素为单位）。
    """


def get_hsv_img(img):
    '''用于处理输入img, 分别返回H,S,V'''
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    row, col = img.shape[0], img.shape[1]

    H = np.zeros([row, col])
    S = np.zeros([row, col])
    V = np.zeros([row, col])
    R = np.zeros([row, col])
    G = np.zeros([row, col])
    B = np.zeros([row, col])

    H, S, V = RGB_HSV_jit(img, H, S, V)

    np.clip(H, 0, 360, out=H)
    np.clip(S, 0, 100, out=S)
    np.clip(V, 0, 100, out=V)

    H[np.where(H < 27)] = 0
    S[np.where(S > 140)] = 0
    V[np.where(V < 30)] = 0

    return H, S, V


@jit(nopython=True)
def RGB_HSI_jit(img):
    img = img.astype(int)
    img_b, img_g, img_r = cv2.split(img)
    row, col = img.shape[0], img.shape[1]
    H = np.zeros([row, col])
    S = np.zeros([row, col])
    I = np.zeros([row, col])

    for i in range(row):
        for j in range(col):
            R = int(img_r[i, j])
            G = int(img_g[i, j])
            B = int(img_b[i, j])
            if R + B + G == 0:
                S[i, j] = 0
            else:
                S[i, j] = 1 - 3 / (R + G + B) * min(R, G, B)
                I[i, j] = (R + G + B) / 3

            R_G = int(np.subtract(img_r[i, j], img_g[i, j]))
            R_B = int(np.subtract(img_r[i, j], img_b[i, j]))
            G_B = int(np.subtract(img_g[i, j], img_b[i, j]))

            # a = 0.5 * (R_G + R_B)
            a = 0.5 * int(R_G + R_B)
            b = math.pow((math.pow(R_G, 2) + np.multiply(R_B, G_B)), 0.5)
            if b == 0:
                b = 0.001

            theta = np.arccos(a / b)
            # if b == 0:
            #     H[i, j] = 255
            # else:
            #     H[i, j] = a / b

            if R_B > R_G:
                H[i, j] = 360 - theta
            elif R_B <= R_G:
                H[i, j] = theta

    return [H, S, I]


def RGB_HSI_ori(img):
    img = img.astype(int)
    img_b, img_g, img_r = cv2.split(img)
    row, col = img.shape[0], img.shape[1]
    H = np.zeros([row, col])
    S = np.zeros([row, col])
    I = np.zeros([row, col])

    for i in range(row):
        for j in range(col):
            R = int(img_r[i, j])
            G = int(img_g[i, j])
            B = int(img_b[i, j])
            if R + B + G == 0:
                S[i, j] = 0
            else:
                S[i, j] = 1 - 3 / (R + G + B) * min(R, G, B)
                I[i, j] = (R + G + B) / 3

            R_G = int(np.subtract(img_r[i, j], img_g[i, j]))
            R_B = int(np.subtract(img_r[i, j], img_b[i, j]))
            G_B = int(np.subtract(img_g[i, j], img_b[i, j]))

            # a = 0.5 * (R_G + R_B)
            a = 0.5 * int(R_G + R_B)
            b = math.pow((math.pow(R_G, 2) + np.multiply(R_B, G_B)), 0.5)
            if b == 0:
                b = 0.001

            theta = np.arccos(a / b)
            # if b == 0:
            #     H[i, j] = 255
            # else:
            #     H[i, j] = a / b

            if R_B > R_G:
                H[i, j] = 360 - theta
            elif R_B <= R_G:
                H[i, j] = theta

    return [H, S, I]


@jit(nopython=True)
def RGB_HSI_jit(img):
    img = img.astype(int)
    img_b, img_g, img_r = cv2.split(img)
    row, col = img.shape[0], img.shape[1]
    H = np.zeros([row, col])
    S = np.zeros([row, col])
    I = np.zeros([row, col])

    for i in range(row):
        for j in range(col):
            R = int(img_r[i, j])
            G = int(img_g[i, j])
            B = int(img_b[i, j])
            if R + B + G == 0:
                S[i, j] = 0
            else:
                S[i, j] = 1 - 3 / (R + G + B) * min(R, G, B)
                I[i, j] = (R + G + B) / 3

            R_G = int(np.subtract(img_r[i, j], img_g[i, j]))
            R_B = int(np.subtract(img_r[i, j], img_b[i, j]))
            G_B = int(np.subtract(img_g[i, j], img_b[i, j]))

            # a = 0.5 * (R_G + R_B)
            a = 0.5 * int(R_G + R_B)
            b = math.pow((math.pow(R_G, 2) + np.multiply(R_B, G_B)), 0.5)
            if b == 0:
                b = 0.001

            theta = np.arccos(a / b)
            # if b == 0:
            #     H[i, j] = 255
            # else:
            #     H[i, j] = a / b

            if R_B > R_G:
                H[i, j] = 360 - theta
            elif R_B <= R_G:
                H[i, j] = theta

    return [H, S, I]


def RGB_HSI(img):
    img = img.astype(int)
    img_b, img_g, img_r = cv2.split(img)
    row, col = img.shape[0], img.shape[1]
    H = np.zeros([row, col])
    S = np.zeros([row, col])
    I = np.zeros([row, col])

    for i in range(row):
        for j in range(col):
            R = int(img_r[i, j])
            G = int(img_g[i, j])
            B = int(img_b[i, j])
            if R + B + G == 0:
                S[i, j] = 0
            else:
                S[i, j] = 1 - 3 / (R + G + B) * min(R, G, B)
                I[i, j] = (R + G + B) / 3

            R_G = int(np.subtract(img_r[i, j], img_g[i, j]))
            R_B = int(np.subtract(img_r[i, j], img_b[i, j]))
            G_B = int(np.subtract(img_g[i, j], img_b[i, j]))

            # a = 0.5 * (R_G + R_B)
            a = 0.5 * int(R_G + R_B)
            b = math.pow((math.pow(R_G, 2) + np.multiply(R_B, G_B)), 0.5)
            if b == 0:
                b = 0.001

            theta = np.arccos(a / b)
            # if b == 0:
            #     H[i, j] = 255
            # else:
            #     H[i, j] = a / b

            if R_B > R_G:
                H[i, j] = 360 - theta
            elif R_B <= R_G:
                H[i, j] = theta

    return [H, S, I]


def RGB_HSV(img, H, S, V):
    '''slow and original method'''
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    row, col = img.shape[0], img.shape[1]
    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

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
                H[i, j] = 0
            elif C_max == R:
                H[i, j] = 60 * ((G - B) / delta)
            elif C_max == G:
                H[i, j] = 60 * ((B - R) / delta + 2)
            elif C_max == B:
                H[i, j] = 60 * ((R - G) / delta + 4)

            if H[i, j] < 0:
                H[i, j] = H[i, j] + 360

            if C_max == 0:
                S[i, j] = 0
            elif C_max != 0:
                S[i, j] = delta / C_max

            V[i, j] = C_max

    return [H, S, V]


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


class tracking():
    def __init__(self):
        self.cord_left_x = -1
        self.cord_left_y = -1
        self.cord_right_x = -1
        self.cord_right_y = -1

    def get_cord_2thread(self, frame, side):
        row, col = frame.shape[0], frame.shape[1]
        img_mask = HSV_process(frame)  ## hsv method of mine
        # cv2.imshow('hsv img', img_mask)

        contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        x, y, w, h = -1, -1, -1, -1
        targetPos_x, targetPos_y = -1, -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # print(area)
            if 20 < area < 300:
                x, y, w, h = cv2.boundingRect(cnt)
                x_total = 0
                y_total = 0

                for i in range(len(cnt[0])):
                    x_total = cnt[0][0][0] + x_total
                    y_total = cnt[0][0][1] + y_total

                targetPos_x = int(x + w / 2)
                targetPos_y = int(y + h / 2)

                cv2.putText(frame, "({:0<2d}, {:0<2d})".format(targetPos_x, targetPos_y), (20, 30),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)  # 文字
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # locol_img = np.zeros([100, 100, 3])

        if targetPos_x == -1 or targetPos_y == -1:
            pass
            # print('not found')
        elif (targetPos_x - 25 > 0 and targetPos_x + 25 < row - 1
              and targetPos_y - 25 > 0 and targetPos_x + 25 < col - 1):
            local_img = frame[targetPos_y - 25:targetPos_y + 25, targetPos_x - 25:targetPos_x + 25, :]
            cv2.imshow('local image', local_img)

            output_img = local_img.copy()

            circles = hough_detect_ball(input_img=local_img, r_min=8,
                                        r_max=15, delta_r=1, num_thetas=60,
                                        bin_threshold=0.4, min_edge_threshold=50,
                                        max_edge_threshold=100)

            for x, y, r, v in circles:
                output_img = cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)

            if len(circles):
                cord_x, cord_y = circles[0][0] + targetPos_x - 25, circles[0][1] + targetPos_y - 25
            elif len(circles) == 0:
                cord_x, cord_y = -1, -1

            if side == 0:
                self.cord_left_x = cord_x
                self.cord_left_y = cord_y
            elif side == 1:
                self.cord_right_x = cord_x
                self.cord_right_y = cord_y

    def return_cord(self):
        return self.cord_left_x, self.cord_left_y, self.cord_right_x, self.cord_right_y


class map():
    def __init__(self):
        self.camera_matrix0 = np.load('D:\DIP-pythonProject\DIP_project\coefficient\cameraMatrix1.npy')
        self.camera_matrix1 = np.load('D:\DIP-pythonProject\DIP_project\coefficient\cameraMatrix2.npy')
        self.distortion0 = np.load('D:\DIP-pythonProject\DIP_project\coefficient\distCoeffs1.npy')
        self.distortion1 = np.load('D:\DIP-pythonProject\DIP_project\coefficient\distCoeffs2.npy')
        self.R = np.load('D:\DIP-pythonProject\DIP_project\coefficient\R.npy')
        self.T = np.load('D:\DIP-pythonProject\DIP_project\coefficient\T.npy')

        self.cx_l = self.camera_matrix0[0, 2]
        self.cx_r = self.camera_matrix1[0, 2]
        self.fx_l = self.camera_matrix0[0, 0]
        self.fx_r = self.camera_matrix1[0, 0]
        self.cy_l = self.camera_matrix0[1, 2]
        self.cy_r = self.camera_matrix1[1, 2]
        self.Tx = self.T[0, 0]

        # 选择是否需要填充
        self.camera_matrix0[0, 2] = self.camera_matrix0[0, 2] + 320
        self.camera_matrix0[1, 2] = self.camera_matrix0[1, 2] + 256
        print("self.camera_matrix0: ")
        print(self.camera_matrix0)
        print()

        # 选择是否需要填充
        self.camera_matrix1[0, 2] = self.camera_matrix1[0, 2] + 320
        self.camera_matrix1[1, 2] = self.camera_matrix1[1, 2] + 256
        print("self.camera_matrix1: ")
        print(self.camera_matrix1)
        print()

        self.HEIGHT = 1024
        self.WIDTH = 1280

        # show_type 决定map的参数，all为能显示全部图像的map；default为原图像的大小，可能会丢失一部分图像
        # 使用时要注意前面camera——matrix的c_x&c_y也要对应的修改
        self.create_correction_map(show_type="all")

    def create_correction_map(self, show_type):

        if show_type == "default":
            self.distortion0[0][0], self.distortion0[0][1], self.distortion0[0][4] = 0, 0, 0
            self.distortion1[0][0], self.distortion1[0][1], self.distortion1[0][4] = 0, 0, 0

            # T[0][0] = -700
            (R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
                cv2.stereoRectify(self.camera_matrix0, self.distortion0, self.camera_matrix1, self.distortion1, \
                                  np.array([self.WIDTH, self.HEIGHT]), self.R,
                                  self.T)  # 计算旋转矩阵和投影矩阵
            (self.map1_l, self.map2_l) = \
                cv2.initUndistortRectifyMap(self.camera_matrix0, self.distortion0, R_l, P_l,
                                            np.array([self.WIDTH, self.HEIGHT]),
                                            cv2.CV_32FC1)  # 计算校正查找映射表
            # 左右图需要分别计算校正查找映射表以及重映射
            (self.map1_r, self.map2_r) = \
                cv2.initUndistortRectifyMap(self.camera_matrix1, self.distortion1, R_r, P_r,
                                            np.array([self.WIDTH, self.HEIGHT]),
                                            cv2.CV_32FC1)

        if show_type == "all":
            self.distortion0[0][0], self.distortion0[0][1], self.distortion0[0][4] = 0, 0, 0
            self.distortion1[0][0], self.distortion1[0][1], self.distortion1[0][4] = 0, 0, 0

            # T[0][0] = -700
            R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2 = \
                cv2.stereoRectify(self.camera_matrix0, self.distortion0,
                                  self.camera_matrix1, self.distortion1,
                                  (int(self.WIDTH * 1.5), int(self.HEIGHT * 1.5)),
                                  self.R, self.T)  # 计算旋转矩阵和投影矩阵

            print("R_l & R_r")
            print(validPixROI1)
            print(validPixROI2)

            # cv2.getOptimalNewCameraMatrix()
            # self.camera_matrix0, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix0, self.distortion0,
            #                                                          imageSize=[self.WIDTH, self.HEIGHT], alpha=1)
            (self.map1_l, self.map2_l) = \
                cv2.initUndistortRectifyMap(self.camera_matrix0, self.distortion0, R_l, P_l,
                                            (int(self.WIDTH * 1.5), int(self.HEIGHT * 1.5)),
                                            cv2.CV_32FC1)  # 计算校正查找映射表
            print("original")
            print(self.camera_matrix1)
            # self.camera_matrix1, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix1, self.distortion1,
            #                                                          imageSize=[self.WIDTH, self.HEIGHT], alpha=1)
            print("after")
            print(self.camera_matrix1)
            # 左右图需要分别计算校正查找映射表以及重映射
            (self.map1_r, self.map2_r) = \
                cv2.initUndistortRectifyMap(self.camera_matrix1, self.distortion1, R_r, P_r,
                                            (int(self.WIDTH * 1.5), int(self.HEIGHT * 1.5)),
                                            cv2.CV_32FC1)

    def polar_correction(self, left_image, right_image):
        self.distortion0[0][0], self.distortion0[0][1], self.distortion0[0][4] = 0, 0, 0
        self.distortion1[0][0], self.distortion1[0][1], self.distortion1[0][4] = 0, 0, 0

        # T[0][0] = -700
        (R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
            cv2.stereoRectify(self.camera_matrix0, self.distortion0, self.camera_matrix1, self.distortion1, \
                              (self.WIDTH, self.HEIGHT), self.R,
                              self.T)  # 计算旋转矩阵和投影矩阵
        # (map1, map2) = \
        #     cv2.initUndistortRectifyMap(self.camera_matrix0, self.distortion0, R_l, P_l, np.array([self.WIDTH, self.HEIGHT]),cv2.CV_32FC1)  # 计算校正查找映射表
        rect_left_image = cv2.remap(left_image, self.map1_l, self.map2_l, cv2.INTER_CUBIC)  # 重映射
        # 左右图需要分别计算校正查找映射表以及重映射
        # (map1, map2) = \
        #     cv2.initUndistortRectifyMap(self.camera_matrix1, self.distortion1, R_r, P_r, np.array([self.WIDTH, self.HEIGHT]), cv2.CV_32FC1)
        rect_right_image = cv2.remap(right_image, self.map1_r, self.map2_r, cv2.INTER_CUBIC)
        return [rect_left_image, rect_right_image]

    def get_3D_coord(self, x_l, y_l, x_r, y_r):
        d = x_l - x_r
        X, Y, Z, W = x_l - self.cx_l, y_l - self.cy_l, self.fx_l, (-d + self.cx_l - self.cx_r) / self.Tx
        x_3d, y_3d, z_3d = X / W, Y / W, Z / W
        return [x_3d, y_3d, z_3d]