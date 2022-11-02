import time
from collections import defaultdict

import cv2
import numpy as np


def detect_table_edge(img, side):
    '''
    用于大致预估桌角的位置
    返回依次为左上，右上，左下，右下的桌角的坐标
    :param img: 输入的图像
    :param side: 左右图像的，左为0，右为1
    :return:
    '''
    img_smooth = cv2.GaussianBlur(img, (5, 5), 7)
    img_edge = cv2.Canny(img_smooth, 130, 250)
    img_shape = img.shape

    minLineLength = 200
    maxLineGap = 10
    lines = cv2.HoughLines(img_edge, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    h, w = img.shape[0], img.shape[1]

    points = []

    if lines is not None:
        for i in range(len(lines)):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + w * (-b))
                y1 = int(y0 + w * a)
                x2 = int(x0 - w * (-b))
                y2 = int(y0 - w * a)
                points.append((x1, y1, x2, y2))
                cv2.line(img, (x1, y1), (x2, y2), (180, 80, 40), 1)

    points_len = len(points)
    cross_point_list = []

    for i in range(points_len):
        for j in range(i + 1, points_len):
            cross_point_list.append(cross_point(points[i], points[j]))

    for point in cross_point_list:
        if point is not None:
            x, y = point
            x = int(x)
            y = int(y)
            if 0 < x < img.shape[1] and 0 < y < img.shape[0]:
                img[y, x] = [20, 225, 170]

    final_list = final_cross(cross_point_list)  ## 调用下面的找交点的函数
    orderd_edge_list = edge_mapping(final_list)

    center = (0, 0)
    for edge in orderd_edge_list:
        x_c, y_c = center
        x_edge, y_edge = edge
        center = (x_edge + x_c, y_edge + y_c)
    x_c, y_c = center
    center = round(x_c / 4), round(y_c / 4)
    print(center)
    count = 1
    for edge in orderd_edge_list:
        cv2.circle(img, edge, 1, (0, 0, 255), 3)
        # cv2.circle(img, center, 1, (0, 255, 255), 3)
        # cv2.line(img, center, edge, (180, 80, 160), 2)
        count = count + 1
    if side == 0:
        cv2.imshow("left edge img", img)
        edge0, edge1, edge2, edge3 = orderd_edge_list
        print('left edges', edge0, edge1, edge2, edge3)
    if side == 1:
        cv2.imshow("right edge img", img)
        edge0, edge1, edge2, edge3 = orderd_edge_list
        print('right edges', edge0, edge1, edge2, edge3)

    return orderd_edge_list


def final_cross(cross_point_list, min_dis=50):
    '''
    用于在hough line获得的直线交点中，算出最有可能时角点的坐标位置
    :param cross_point_list:输入为交点的list
    :param min_dis:用于选定kernel的范围， eg：min—_dis=50为100*100的范围
    :return:edges_list：投票数最多的四个角点
    '''
    voter = defaultdict(int)
    for point in cross_point_list:
        if point is not None:
            x, y = point
            tmp_x_list = []
            tmp_y_list = []
            for tmp_point in cross_point_list:
                if tmp_point is not None:
                    tmp_x, tmp_y = tmp_point
                    if abs(x - tmp_x) < min_dis and abs(y - tmp_y) < min_dis:
                        tmp_x_list.append(tmp_x)
                        tmp_y_list.append(tmp_y)

            x_avg = np.mean(tmp_x_list)
            y_avg = np.mean(tmp_y_list)
            voter[x_avg, y_avg] = voter[x_avg, y_avg] + 1

    new_list = sorted(voter.items(), key=lambda i: -i[1])
    # print('new list', new_list)
    edge0, votes = new_list[0]
    edge1, votes = new_list[1]
    edge2, votes = new_list[2]
    edge3, votes = new_list[3]

    edge0 = (int(edge0[0]), int(edge0[1]))
    edge1 = (int(edge1[0]), int(edge1[1]))
    edge2 = (int(edge2[0]), int(edge2[1]))
    edge3 = (int(edge3[0]), int(edge3[1]))
    edges_list = (edge0, edge1, edge2, edge3)

    # print('edges', edge0, edge1, edge2, edge3)

    return edges_list


def precise_table_edge(img_copy, orderd_edge_list, local_edge=15, min_difference=5):
    '''
    :param img_copy: 复制的图像
    :param orderd_edge_list: 左上，右上，左下，右下的估计坐标
    :return:
    '''
    local_edge = 15
    local_image_size = 2 * local_edge  # 区域的大小为 2*local_edge+1
    min_difference = 5
    local_image = np.zeros((local_image_size, local_image_size), np.uint8)

    x_list = []
    y_list = []

    if len(orderd_edge_list) == 4:
        for edge in orderd_edge_list:
            x, y = edge
            x_list.append(x)
            y_list.append(y)

    # case 0：左上角
    local_image_rgb0 = img_copy[y_list[0] - local_edge:y_list[0] + local_edge,
                       x_list[0] - local_edge:x_list[0] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb0, cv2.COLOR_BGR2GRAY)
    retval, local_img0 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img0 == 255)

    dis_min = 9999
    wanted_index_0 = -1, -1  # 左上
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if x + y <= dis_min and abs(abs(x) - abs(y)) <= min_difference:
            dis_min = x + y
            wanted_index_0 = x, y

    # case 1：右上角
    local_image_rgb1 = img_copy[y_list[1] - local_edge:y_list[1] + local_edge,
                       x_list[1] - local_edge:x_list[1] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb1, cv2.COLOR_BGR2GRAY)
    retval, local_img1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img1 == 255)

    dis_min = 9999
    wanted_index_1 = -1, -1  # 右上
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if - x + y <= dis_min and abs(x + y - local_image_size - 1) <= min_difference:
            dis_min = - x + y
            wanted_index_1 = x, y

    # case 2：左下角
    local_image_rgb2 = img_copy[y_list[2] - local_edge:y_list[2] + local_edge,
                       x_list[2] - local_edge:x_list[2] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb2, cv2.COLOR_BGR2GRAY)
    retval, local_img2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img2 == 255)

    dis_min = 9999
    wanted_index_2 = -1, -1  # 左下
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if x - y <= dis_min and abs(x + y - local_image_size - 1) <= min_difference:
            dis_min = x - y
            wanted_index_2 = x, y

    # case 3：右下角
    local_image_rgb3 = img_copy[y_list[3] - local_edge:y_list[3] + local_edge,
                       x_list[3] - local_edge:x_list[3] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb3, cv2.COLOR_BGR2GRAY)
    retval, local_img3 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img3 == 255)

    dis_min = 9999
    wanted_index_3 = -1, -1  # 右下
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if - x - y <= dis_min and abs(abs(x) - abs(y)) <= min_difference:
            dis_min = - x - y
            wanted_index_3 = x, y

    edge0, edge1, edge2, edge3 = (-1, -1), (-1, -1), (-1, -1), (-1, -1)

    # case 0
    local_x, local_y = wanted_index_0
    if local_x != -1 and local_y != -1:
        edge0 = (x_list[0] + local_x - local_edge, y_list[0] + local_y - local_edge)

    # case 1
    (local_x, local_y) = wanted_index_1
    if local_x != -1 and local_y != -1:
        edge1 = (x_list[1] + local_x - local_edge, y_list[1] + local_y - local_edge)

    # case 2
    local_x, local_y = wanted_index_2
    if local_x != -1 and local_y != -1:
        edge2 = (x_list[2] + local_x - local_edge, y_list[2] + local_y - local_edge)

    # case 3
    local_x, local_y = wanted_index_3
    if local_x != -1 and local_y != -1:
        edge3 = (x_list[3] + local_x - local_edge, y_list[3] + local_y - local_edge)

    precise_edges = edge0, edge1, edge2, edge3

    return precise_edges


def find_precise_edge(img_copy, orderd_edge_list, local_edge = 15):
    '''
    无imshow版本
    :param img_copy: 复制的图像
    :param orderd_edge_list: 左上，右上，左下，右下的估计坐标
    :return:
    '''
    local_image_size = 2 * 15  # 区域的大小为 2*local_edge
    local_image = np.zeros((local_image_size, local_image_size), np.uint8)

    x_list = []
    y_list = []

    if len(orderd_edge_list) == 4:
        for edge in orderd_edge_list:
            x, y = edge
            x_list.append(x)
            y_list.append(y)

    # case 0：左上角
    # print("左上角区域", y_list[0] - local_edge, y_list[0] + local_edge, x_list[0] - local_edge, x_list[0] + local_edge)
    local_image_rgb0 = img_copy[y_list[0] - local_edge:y_list[0] + local_edge,
                       x_list[0] - local_edge:x_list[0] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb0, cv2.COLOR_BGR2GRAY)
    retval, local_img0 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img0 == 255)
    # print(type(indexs_white_part))
    # print("indexs_white_part", indexs_white_part)

    min_difference = 5
    dis_min = 9999
    wanted_index_0 = -1, -1  # 左上
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if x + y <= dis_min and abs(abs(x) - abs(y)) <= min_difference:
            dis_min = x + y
            wanted_index_0 = x, y

    # case 1：右上角
    # print("右上角区域", y_list[1] - local_edge, y_list[1] + local_edge, x_list[1] - local_edge, x_list[1] + local_edge)
    local_image_rgb1 = img_copy[y_list[1] - local_edge:y_list[1] + local_edge,
                       x_list[1] - local_edge:x_list[1] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb1, cv2.COLOR_BGR2GRAY)
    retval, local_img1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img1 == 255)

    dis_min = 9999
    wanted_index_1 = -1, -1  # 右上
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if - x + y <= dis_min and abs(x + y - local_image_size - 1) <= min_difference:
            dis_min = - x + y
            wanted_index_1 = x, y

    # case 2：左下角
    # print("左下角区域", y_list[2] - local_edge, y_list[2] + local_edge, x_list[2] - local_edge, x_list[2] + local_edge)
    local_image_rgb2 = img_copy[y_list[2] - local_edge:y_list[2] + local_edge,
                       x_list[2] - local_edge:x_list[2] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb2, cv2.COLOR_BGR2GRAY)
    retval, local_img2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img2 == 255)

    dis_min = 9999
    wanted_index_2 = -1, -1  # 左下
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if x - y <= dis_min and abs(x + y - local_image_size - 1) <= min_difference:
            dis_min = x - y
            wanted_index_2 = x, y


    # case 3：右下角
    # print("右下角区域", y_list[3] - local_edge, y_list[3] + local_edge, x_list[3] - local_edge, x_list[3] + local_edge)
    local_image_rgb3 = img_copy[y_list[3] - local_edge:y_list[3] + local_edge,
                       x_list[3] - local_edge:x_list[3] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb3, cv2.COLOR_BGR2GRAY)
    retval, local_img3 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img3 == 255)

    dis_min = 9999
    wanted_index_3 = -1, -1  # 右下
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if - x - y <= dis_min and abs(abs(x) - abs(y)) <= min_difference:
            dis_min = - x - y
            wanted_index_3 = x, y

    edge0, edge1, edge2, edge3 = 0, 0, 0, 0

    # case 0
    local_x, local_y = wanted_index_0
    if local_x != -1 and local_y != -1:
        # img_copy[y_list[0] + local_y, x_list[0] + local_x] = [255, 0 , 0]
        edge0 = (x_list[0] + local_x - local_edge, y_list[0] + local_y - local_edge)

    # case 1
    (local_x, local_y) = wanted_index_1
    if local_x != -1 and local_y != -1:
        # img_copy[y_list[0] + local_y, x_list[0] + local_x] = [255, 0 , 0]
        edge1 = (x_list[1] + local_x - local_edge, y_list[1] + local_y - local_edge)

    # case 2
    local_x, local_y = wanted_index_2
    if local_x != -1 and local_y != -1:
        # img_copy[y_list[0] + local_y, x_list[0] + local_x] = [255, 0 , 0]
        edge2 = (x_list[2] + local_x - local_edge, y_list[2] + local_y - local_edge)

    # case 3
    local_x, local_y = wanted_index_3
    if local_x != -1 and local_y != -1:
        # img_copy[y_list[0] + local_y, x_list[0] + local_x] = [255, 0 , 0]
        edge3 = (x_list[3] + local_x - local_edge, y_list[3] + local_y - local_edge)

    precise_edges = [edge0, edge1, edge2, edge3]
    return precise_edges


def find_precise_edge_and_show(img_copy, orderd_edge_list, local_edge=15, min_difference=2):
    '''
    有imshow版本
    :param img_copy: 复制的图像
    :param orderd_edge_list: 左上，右上，左下，右下的估计坐标
    :return:
    '''
    local_edge = 15
    local_image_size = 2 * local_edge  # 区域的大小为 2*local_edge+1
    local_image = np.zeros((local_image_size, local_image_size), np.uint8)

    x_list = []
    y_list = []

    if len(orderd_edge_list) == 4:
        for edge in orderd_edge_list:
            x, y = edge
            x_list.append(x)
            y_list.append(y)

    # case 0：左上角
    print("左上角区域", y_list[0] - local_edge, y_list[0] + local_edge, x_list[0] - local_edge, x_list[0] + local_edge)
    local_image_rgb0 = img_copy[y_list[0] - local_edge:y_list[0] + local_edge,
                       x_list[0] - local_edge:x_list[0] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb0, cv2.COLOR_BGR2GRAY)
    retval, local_img0 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img0 == 255)
    # print(type(indexs_white_part))
    # print("indexs_white_part", indexs_white_part)

    min_difference = 5
    dis_min = 9999
    wanted_index_0 = -1, -1  # 左上
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if x + y <= dis_min and abs(abs(x) - abs(y)) <= min_difference:
            dis_min = x + y
            wanted_index_0 = x, y

    # case 1：右上角
    print("右上角区域", y_list[1] - local_edge, y_list[1] + local_edge, x_list[1] - local_edge, x_list[1] + local_edge)
    local_image_rgb1 = img_copy[y_list[1] - local_edge:y_list[1] + local_edge,
                       x_list[1] - local_edge:x_list[1] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb1, cv2.COLOR_BGR2GRAY)
    retval, local_img1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img1 == 255)

    print(indexs_white_part)

    dis_min = 9999
    wanted_index_1 = -1, -1  # 右上
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if - x + y <= dis_min and abs(x + y - local_image_size - 1) <= min_difference:
            dis_min = - x + y
            wanted_index_1 = x, y

    # case 2：左下角
    print("左下角区域", y_list[2] - local_edge, y_list[2] + local_edge, x_list[2] - local_edge, x_list[2] + local_edge)
    local_image_rgb2 = img_copy[y_list[2] - local_edge:y_list[2] + local_edge,
                       x_list[2] - local_edge:x_list[2] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb2, cv2.COLOR_BGR2GRAY)
    retval, local_img2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img2 == 255)

    dis_min = 9999
    wanted_index_2 = -1, -1  # 左下
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if x - y <= dis_min and abs(x + y - local_image_size - 1) <= min_difference:
            dis_min = x - y
            wanted_index_2 = x, y
    print("dismin", dis_min)

    # case 3：右下角
    print("右下角区域", y_list[3] - local_edge, y_list[3] + local_edge, x_list[3] - local_edge, x_list[3] + local_edge)
    local_image_rgb3 = img_copy[y_list[3] - local_edge:y_list[3] + local_edge,
                       x_list[3] - local_edge:x_list[3] + local_edge]
    img_gray = cv2.cvtColor(local_image_rgb3, cv2.COLOR_BGR2GRAY)
    retval, local_img3 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    indexs_white_part = np.argwhere(local_img3 == 255)

    dis_min = 9999
    wanted_index_3 = -1, -1  # 右下
    for index_xy_pair in indexs_white_part:
        y, x = index_xy_pair
        if - x - y <= dis_min and abs(abs(x) - abs(y)) <= min_difference:
            dis_min = - x - y
            wanted_index_3 = x, y

    cv2.imshow("local_img1", local_img1)
    cv2.imshow("local_img2", local_img2)
    cv2.imshow("local_img3", local_img3)
    cv2.imshow("local_img0", local_img0)

    print("wanted_index0", wanted_index_0)
    print("wanted_index1", wanted_index_1)
    print("wanted_index2", wanted_index_2)
    print("wanted_index3", wanted_index_3)

    # local_img0_res = cv2.cvtColor(local_img0, cv2.COLOR_GRAY2BGR)
    # local_img1_res = cv2.cvtColor(local_img1, cv2.COLOR_GRAY2BGR)
    # local_img2_res = cv2.cvtColor(local_img2, cv2.COLOR_GRAY2BGR)
    # local_img3_res = cv2.cvtColor(local_img3, cv2.COLOR_GRAY2BGR)

    edge0, edge1, edge2, edge3 = (-1, -1), (-1, -1), (-1, -1), (-1, -1)

    # case 0
    local_x, local_y = wanted_index_0
    if local_x != -1 and local_y != -1:
        # img_copy[y_list[0] + local_y, x_list[0] + local_x] = [255, 0 , 0]
        edge0 = (x_list[0] + local_x - local_edge, y_list[0] + local_y - local_edge)
        cv2.circle(local_image_rgb0, wanted_index_0, 1, (0, 255, 0), 2)
        cv2.circle(img_copy, edge0, 1, (0, 0, 255), 3)

    print("case 0", local_x - local_edge, local_y - local_edge)
    # cv2.imshow("copy", img_copy)
    cv2.imshow("local_img0_res", local_image_rgb0)

    # case 1
    (local_x, local_y) = wanted_index_1
    if local_x != -1 and local_y != -1:
        # img_copy[y_list[0] + local_y, x_list[0] + local_x] = [255, 0 , 0]
        edge1 = (x_list[1] + local_x - local_edge, y_list[1] + local_y - local_edge)
        cv2.circle(local_image_rgb1, wanted_index_1, 1, (0, 255, 0), 2)
        cv2.circle(img_copy, edge1, 1, (0, 0, 255), 3)

    print("case 1", local_x - local_edge, local_y - local_edge)
    # cv2.imshow("copy", img_copy)
    cv2.imshow("local_img1_res", local_image_rgb1)

    # case 2
    local_x, local_y = wanted_index_2
    if local_x != -1 and local_y != -1:
        # img_copy[y_list[0] + local_y, x_list[0] + local_x] = [255, 0 , 0]
        edge2 = (x_list[2] + local_x - local_edge, y_list[2] + local_y - local_edge)
        cv2.circle(local_image_rgb2, wanted_index_2, 1, (0, 255, 0), 2)
        cv2.circle(img_copy, edge2, 1, (0, 0, 255), 3)

    print("case 2", local_x - local_edge, local_y - local_edge)
    # cv2.imshow("copy", img_copy)
    cv2.imshow("local_img2_res", local_image_rgb2)

    # case 3
    local_x, local_y = wanted_index_3
    if local_x != -1 and local_y != -1:
        # img_copy[y_list[0] + local_y, x_list[0] + local_x] = [255, 0 , 0]
        edge3 = (x_list[3] + local_x - local_edge, y_list[3] + local_y - local_edge)
        cv2.circle(local_image_rgb3, wanted_index_3, 1, (0, 255, 0), 2)
        cv2.circle(img_copy, edge3, 1, (0, 0, 255), 3)

    print("case 3", local_x - local_edge, local_y - local_edge)
    # cv2.imshow("copy", img_copy)
    cv2.imshow("local_img3_res", local_image_rgb3)

    down_width = int(img_copy.shape[1] * 0.6)
    down_height = int(img_copy.shape[0] * 0.6)
    down_points = (down_width, down_height)
    img_copy_resize = cv2.resize(img_copy, down_points, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("img_copy_resize", img_copy_resize)

    precise_edges = edge0, edge1, edge2, edge3

    return precise_edges


def edge_mapping(edges_list):
    left_top_edge, right_top_edge, left_bot_edge, right_bot_edge = (0, 0), (0, 0), (0, 0), (0, 0)
    center = (0, 0)
    for edge in edges_list:
        x_c, y_c = center
        x_edge, y_edge = edge
        center = (x_edge + x_c, y_edge + y_c)
    x_c, y_c = center
    center = round(x_c / 4), round(y_c / 4)
    # print(center)

    if len(edges_list) != 4:
        return None
    for edge in edges_list:
        tmp_x, tmp_y = edge
        x_c, y_c = center
        x_vector = tmp_x - x_c
        y_vector = tmp_y - y_c
        if x_vector < 0 and y_vector < 0:
            left_top_edge = edge
        if x_vector > 0 and y_vector < 0:
            right_top_edge = edge
        if x_vector < 0 and y_vector > 0:
            left_bot_edge = edge
        if x_vector > 0 and y_vector > 0:
            right_bot_edge = edge
    orderd_edge_list = left_top_edge, right_top_edge, left_bot_edge, right_bot_edge
    return orderd_edge_list


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
    img = cv2.imread("frame2.png")
    img_shape = img.shape
    # img_copy = np.zeros(img_shape, np.uint8)
    img_copy = img.copy()
    orderd_edge_list = detect_table_edge(img, 0)
    # find_edge_local(img_copy, orderd_edge_list)
    # print('edge list:', edge_list)

    precise_edges = find_precise_edge(img_copy, orderd_edge_list, local_edge = 15)

    print("preceis edges", precise_edges)

    cv2.line(img, precise_edges[0], precise_edges[3], (40, 240, 0), 1)
    cv2.line(img, precise_edges[1], precise_edges[2], (40, 240, 0), 1)

    line1 = [precise_edges[0][0],precise_edges[0][1],precise_edges[3][0],precise_edges[3][1]]
    line2 = [precise_edges[1][0],precise_edges[1][1],precise_edges[2][0], precise_edges[2][1]]

    cross_point = cross_point(line1, line2)
    x_cro, y_cro = round(cross_point[0]), round(cross_point[1])
    cv2.circle(img, (x_cro, y_cro), 3, (15, 210, 160), 4)

    print("center:" ,(x_cro, y_cro))

    down_width = int(img.shape[1] * 0.6)
    down_height = int(img.shape[0] * 0.6)
    down_points = (down_width, down_height)
    img_copy_resize = cv2.resize(img, down_points, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("img_copy_resize", img_copy_resize)

    cv2.waitKey(0)
