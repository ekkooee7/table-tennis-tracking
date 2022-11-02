'''
author: ekko
python version: 3.8

可用于双线程处理左右摄像头输入图像

利用return_cord函数，获得当前球坐标
'''
import time

import cv2


from DIP_project.IMAGE_PROCESS import HSV_process, hough_detect_ball


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
            print('not found')
            self.refresh_cord()  # 刷新输出坐标
            pass

        elif (targetPos_x - 25 > 0 and targetPos_x + 25 < row - 1
              and targetPos_y - 25 > 0 and targetPos_x + 25 < col - 1):
            local_img = frame[targetPos_y - 25:targetPos_y + 25, targetPos_x - 25:targetPos_x + 25, :]
            cv2.imshow('local image', local_img)

            output_img = local_img.copy()

            t0 = time.time()
            circles = hough_detect_ball(input_img=local_img, r_min=8,
                                        r_max=15, delta_r=1, num_thetas=60,
                                        bin_threshold=0.4, min_edge_threshold=50,
                                        max_edge_threshold=100)
            t1 = time.time()
            print("hough time:",t1-t0)

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



    def refresh_cord(self):
        self.cord_left_x = -1
        self.cord_left_y = -1
        self.cord_right_x = -1
        self.cord_right_y = -1


    def return_cord(self):
        return self.cord_left_x, self.cord_left_y, self.cord_right_x, self.cord_right_y