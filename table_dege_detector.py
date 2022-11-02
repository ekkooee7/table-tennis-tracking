import time

import cv2
import numpy as np


def empty(a):
    pass


def cross_point(line1, line2):  # 计算交点函数
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        if k1 == k2:
            return None
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


if __name__ == '__main__':
    img = cv2.imread("table.png")

    img_smooth = cv2.GaussianBlur(img, (5, 5), 7)
    img_edge = cv2.Canny(img_smooth, 200, 250)
    cv2.imshow("1", img_edge)

    # img_shape = (img.shape[0], img.shape[1])
    img_shape = img.shape
    h, w = img.shape[0], img.shape[1]
    print(img_shape)

    img_blank = np.zeros(img_shape)

    img_blank_0 = np.zeros(img_shape)

    # contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img_blank, contours, -1, (255, 0, 0), 3)

    lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 200)

    points = []

    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + w * (-b))
            y1 = int(y0 + w * (a))
            x2 = int(x0 - w * (-b))
            y2 = int(y0 - w * (a))

            points.append((x1, y1, x2, y2))

            cv2.line(img_blank, (x1, y1), (x2, y2), (10, 0, 0), 1)

    points_len = len(points)
    cross_point_list = []
    print(points_len)
    for i in range(points_len):
        for j in range(i + 1, points_len):
            print(i, j)
            cross_point_list.append(cross_point(points[i], points[j]))

    print(cross_point_list)

    for point in cross_point_list:
        if point is not None:
            x, y = point
            x = int(x)
            y = int(y)
            if 0 < x < img.shape[0] and 0 < y < img.shape[1]:
                print(y, x)

                img_blank_0[y, x] = [255, 255, 255]

    # dst = cv2.cornerHarris(img_edge, 2, 3, 0.2)
    # a = dst > 0.01 * dst.max()
    # img[a] = [0, 0, 255]
    # cv2.imshow("2", dst)

    cv2.imshow("3", img)

    cv2.imshow("blank", img_blank)

    cv2.imshow("blank_0", img_blank_0)

    cv2.waitKey(0)
