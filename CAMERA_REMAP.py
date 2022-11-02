'''
author: ekko
python version: 3.8

CAMERA_REMAP 用于相机的reconstruction
读取利用opencv标定的双目相机的参数

create_correction_map 函数用在相机初始化阶段
其中
parameter “default” 为 opencv标准的相机矫正，可能造成图像部分缺失
parameter “all” 会恢复整张图像，但是会在矫正前填充图像，造成更大运算量

因为对图像做了填充

'''
import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from numba import jit
from collections import defaultdict


class CAMERA_REMAP():
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
        '''
        函数用在相机初始化阶段
        :param show_type: “default” 为opencv标准的相机矫正，可能造成图像部分缺失
                          “all” 会恢复整张图像，但是会在矫正前填充图像，造成更大运算量
        :return: 无返回值，运行该函数会跟新class中相关的图像校正矩阵等变量
        '''
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
        '''
        图像矫正函数
        :param left_image: 左图像输入
        :param right_image: 右图像输入
        :return: 左校正图像， 右校正图像
        '''
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
        '''
        用于根据左右相机获得的二位坐标，获得三维重构的坐标
        :param x_l: 左相机中球的x坐标（水平）
        :param y_l: 左相机中球的y坐标（垂直）（以左相机建立坐标系）
        :param x_r: 右相机中球的x坐标（水平）
        :param y_r: 右相机中球的y坐标（垂直）（未用到）
        :return:
        '''
        d = x_l - x_r
        X, Y, Z, W = x_l - self.cx_l, y_l - self.cy_l, self.fx_l, (-d + self.cx_l - self.cx_r) / self.Tx
        x_3d, y_3d, z_3d = X / W, Y / W, Z / W
        return [x_3d, y_3d, z_3d]