import math
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import COLOR_library


def empty(a):
    pass


def draw_direction(img, lx, ly, nx, ny):
    # 根据上一位置与当前位置计算移动方向并绘制箭头
    dx = nx - lx
    dy = ny - ly
    if abs(dx) < 2 and abs(dy) < 2:
        dx = 0
        dy = 0
    else:
        r = (dx ** 2 + dy ** 2) ** 0.5
        dx = int(dx / r * 40)
        dy = int(dy / r * 40)
        # print(dx, dy)
    cv2.arrowedLine(img, (60, 100), (60 + dx, 100 + dy), (0, 255, 0), 2)


def get_ball(img):
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    imgMask = cv2.inRange(imgHsv, lower, upper)  # 获取遮罩
    imgOutput = cv2.bitwise_and(img, img, mask=imgMask)
    contours, hierarchy = cv2.findContours(imgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 查找轮廓
    # CV_RETR_EXTERNAL 只检测最外围轮廓
    # CV_CHAIN_APPROX_NONE 保存物体边界上所有连续的轮廓点到contours向量内
    imgMask = cv2.cvtColor(imgMask, cv2.COLOR_GRAY2BGR)  # 转换后，后期才能够与原画面拼接，否则与原图维数不同

    # 下面的代码查找包围框，并绘制
    x, y, w, h = -1, -1, -1, -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if 50 < area < 300:
            x, y, w, h = cv2.boundingRect(cnt)
            x_total = 0
            y_total = 0

            # newimage = img[y + 2:y + h - 2, x + 2:x + w - 2]  # 先用y确定高，再用x确定宽

            ##############################################################################
            # method 1
            ##############################################################################
            # img_x = []
            # img_y = []
            # for i in range(x + 2, x + w - 2):
            #     for j in range(y + 2, y + h - 2):
            #
            #         if img[j, i, 1] >= 5:
            #             img_x.append(i)
            #             img_y.append(j)
            #
            # targetPos_x = round(np.mean(img_x))
            # targetPos_y = round(np.mean(img_y))
            #############################################################################
            # method 2
            # #############################################################################
            for i in range(len(cnt[0])):
                x_total = cnt[0][0][0] + x_total
                y_total = cnt[0][0][1] + y_total

            targetPos_x = int(x + w / 2)
            targetPos_y = int(y + h / 2)

            cv2.putText(img, "({:0<2d}, {:0<2d})".format(targetPos_x, targetPos_y), (20, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)  # 文字
            # draw_direction(img, lastPos_x, lastPos_y, targetPos_x, targetPos_y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # imgStack = np.hstack([img, imgOutput])
            # # imgStack = np.hstack([img, imgMask])            # 拼接
            # cv2.imshow('origin_left', img)
            # cv2.imshow('Horizontal Stacking left', imgOutput)  # 显示

    # gray = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2GRAY)
    # _, threshed_img = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    return imgOutput


if __name__ == '__main__':
    video_path = "videos/ball_left_0_406.avi"
    cap = cv2.VideoCapture(video_path)
    count = 0
    t0 = time.time()

    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 640, 300)
    cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
    cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
    cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
    cv2.createTrackbar("HUE Max", "HSV", 0, 179, empty)
    cv2.createTrackbar("SAT Max", "HSV", 0, 255, empty)
    cv2.createTrackbar("VALUE Max", "HSV", 0, 255, empty)

    time.sleep(1)

    while cap.isOpened():  # for 'q' key退出
        ret, frame = cap.read()

        row, col = frame.shape[0], frame.shape[1]
        cv2.imshow("frame", frame)

        t1 = time.time()
        if count == 100:
            print("frame/s = " + str(round(100 / (t1 - t0), 3)))
            t0 = t1
            count = 0
        else:
            count = count + 1

        # cv2.imshow("frame", frame)

        img_mask = COLOR_library.HSV_process(frame)  ## hsv method of mine

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
        elif (targetPos_x - 25 > 0 and targetPos_x + 25 < row-1
                    and targetPos_y - 25 > 0 and targetPos_x + 25 < col-1):
            local_img=frame[targetPos_y-25:targetPos_y+25,targetPos_x-25:targetPos_x+25,:]
            cv2.imshow('local image', local_img)

        img_1 = get_ball(frame)
        cv2.imshow("origin image", frame)
        cv2.imshow("HSV image", img_mask)
        cv2.imshow("hsv parameter test", img_1)

        key = cv2.waitKey(5) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)
