'''
author: ekko
python version: 3.8



'''

import cv2
import numpy as np
from numba import jit
from collections import defaultdict


def hough_detect_ball(input_img, r_min=8, r_max=20, delta_r=1, num_thetas=30,
                      bin_threshold=0.4, min_edge_threshold=50, max_edge_threshold=100):
    '''
    用于检测球
    输入参数包括球相关的半径信息;
    霍夫圆检测相关的num_thetas等;
    canny边缘检测用到的edge_threshold
    调用 “find_hough_circles_fast” 函数
    :param input_img: 输入图像
    :param r_min: 霍夫圆检测最小半径r_min
    :param r_max: 霍夫圆检测最大半径r_max
    :param delta_r: 霍夫圆检测delta_r
    :param num_thetas: 霍夫圆检测num_thetas
    :param bin_threshold:
    :param min_edge_threshold:
    :param max_edge_threshold:
    :return:
    '''

    # edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    edge_image = cv2.Canny(input_img, min_edge_threshold, max_edge_threshold)

    delta_theta = int(360 / num_thetas)

    if edge_image is not None:
        circles = find_hough_circles_fast(edge_image, r_min, r_max, delta_r, num_thetas,
                                          bin_threshold)
        # circles = Hough_circle_detection(edge_image, r_min, r_max, delta_r, delta_theta, bin_threshold, 6)

    return circles

def Hough_circle_detection(edge_img, r_min, r_max, delta_r, delta_theta, thresh_hold, min_dis, return_type='return_all'):
    height, width = edge_img.shape[0], edge_img.shape[1]
    thetas = np.arange(0, 360, step=delta_theta)
    rs = np.arange(r_min, r_max, delta_r)
    theta_num = int(360 / delta_theta)

    circle_candidates = []
    for r in rs:
        for i in range(theta_num):
            x = int(np.cos(np.deg2rad(thetas[i])))
            y = int(np.sin(np.deg2rad(thetas[i])))
            circle_candidates.append((r, x, y))

    vote_nums = defaultdict(int)

    for j in range(width):
        for i in range(height):
            if edge_img[i][j] != 0:
                for r, dis_x, dis_y in circle_candidates:
                    center_x = i - dis_x
                    center_y = j - dis_y
                    vote_nums[(r, center_x, center_y)] = vote_nums[(r, center_x, center_y)] + 1

    out_circles = []

    if return_type == 'return_max':
        new_list = sorted(vote_nums.items(), key=lambda i: -i[1])
        candidate_circle, votes = new_list[0]
        r, x, y = candidate_circle
        current_vote_percentage = votes / theta_num
        if current_vote_percentage >= thresh_hold:
            out_circles.append((x, y, r, current_vote_percentage))

    elif return_type == 'return_all':
        for candidate_circle, votes in sorted(vote_nums.items(), key=lambda i: -i[1]):
            r, x, y = candidate_circle
            current_vote_percentage = votes / theta_num
            if current_vote_percentage >= thresh_hold:
                out_circles.append((x, y, r, current_vote_percentage))

    postprocess_circles = []
    for x, y, r, v in out_circles:
        for xc, yc, rc, vc in postprocess_circles:
            if abs(x - xc) > min_dis or abs(y - yc) > min_dis or abs(r - rc) > min_dis:
                postprocess_circles.append((x, y, r, v))

    out_circles = postprocess_circles

    return out_circles


def find_hough_circles_fast(edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process=True):
    img_height, img_width = edge_image.shape[:2]

    dtheta = int(360 / num_thetas)

    thetas = np.arange(0, 360, step=dtheta)

    rs = np.arange(r_min, r_max, step=delta_r)

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    circle_candidates = []
    for r in rs:
        for t in range(num_thetas):
            circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))

    accumulator = defaultdict(int)


    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0:
                for r, rcos_t, rsin_t in circle_candidates:
                    x_center = x - rcos_t
                    y_center = y - rsin_t
                    accumulator[(x_center, y_center, r)] += 1
    out_circles = []

    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y, r = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold:
            out_circles.append((x, y, r, current_vote_percentage))

    if post_process:
        pixel_threshold = 5
        postprocess_circles = []
        for x, y, r, v in out_circles:

            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for
                   xc, yc, rc, v in postprocess_circles):
                postprocess_circles.append((x, y, r, v))
        out_circles = postprocess_circles

    return out_circles

# def find_hough_circles_fast(edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process=True):
#     # image size
#     img_height, img_width = edge_image.shape[:2]
#
#     # R and Theta ranges
#     dtheta = int(360 / num_thetas)
#
#     ## Thetas is bins created from 0 to 360 degree with increment of the dtheta
#     thetas = np.arange(0, 360, step=dtheta)
#
#     ## Radius ranges from r_min to r_max
#     rs = np.arange(r_min, r_max, step=delta_r)
#
#     # Calculate Cos(theta) and Sin(theta) it will be required later
#     cos_thetas = np.cos(np.deg2rad(thetas))
#     sin_thetas = np.sin(np.deg2rad(thetas))
#
#     # Evaluate and keep ready the candidate circles dx and dy for different delta radius
#     # based on the the parametric equation of circle.
#     # x = x_center + r * cos(t) and y = y_center + r * sin(t),
#     # where (x_center,y_center) is Center of candidate circle with radius r. t in range of [0,2PI)
#     circle_candidates = []
#     for r in rs:
#         for t in range(num_thetas):
#             # instead of using pre-calculated cos and sin theta values you can calculate here itself by following
#             # circle_candidates.append((r, int(r*cos(2*pi*t/num_thetas)), int(r*sin(2*pi*t/num_thetas))))
#             # but its better to pre-calculate and use it here.
#             circle_candidates.append((r, int(r * cos_thetas[t]), int(r * sin_thetas[t])))
#
#     # Hough Accumulator, we are using defaultdic instead of standard dict as this will initialize for key which is not
#     # aready present in the dictionary instead of throwing exception.
#     accumulator = defaultdict(int)
#
#     # accumulator = vote(edge_image, accumulator, circle_candidates, img_height, img_width)
#
#     for y in range(img_height):
#         for x in range(img_width):
#             if edge_image[y][x] != 0:  # white pixel
#                 # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
#                 for r, rcos_t, rsin_t in circle_candidates:
#                     x_center = x - rcos_t
#                     y_center = y - rsin_t
#                     accumulator[(x_center, y_center, r)] += 1  # vote for current candidate
#
#     # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold)
#     out_circles = []
#
#     for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
#         x, y, r = candidate_circle
#         current_vote_percentage = votes / num_thetas
#         if current_vote_percentage >= bin_threshold:
#             # Shortlist the circle for final result
#             out_circles.append((x, y, r, current_vote_percentage))
#             # print(x, y, r, current_vote_percentage)
#
#     # Post process the results, can add more post processing later.
#     if post_process:
#         pixel_threshold = 5
#         postprocess_circles = []
#         for x, y, r, v in out_circles:
#             # Exclude circles that are too close of each other
#             # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
#             # Remove nearby duplicate circles based on pixel_threshold
#             if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold or abs(r - rc) > pixel_threshold for
#                    xc, yc, rc, v in postprocess_circles):
#                 postprocess_circles.append((x, y, r, v))
#         out_circles = postprocess_circles
#
#     return out_circles


def HSV_process(img):
    '''用于处理输入img， 输出给定的HSV范围内的RGB图像'''
    # img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    row, col = img.shape[0], img.shape[1]

    H = np.zeros([row, col])
    S = np.zeros([row, col])
    V = np.zeros([row, col])

    H, S, V = RGB_HSV_jit(img, H, S, V)

    img_hsv = np.zeros([row, col, 3])

    img_hsv[:, :, 0] = H
    img_hsv[:, :, 1] = S
    img_hsv[:, :, 2] = V

    lower = np.array([9, 184, 202])
    upper = np.array([66, 255, 255])

    # 返回黄色区域的二值图像
    img_hsv = cv2.inRange(img_hsv, lower, upper)

    return img_hsv


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