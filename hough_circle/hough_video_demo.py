import math
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from numba import jit

from DIP_project import COLOR_library
from DIP_project import TRACKING
from DIP_project.hough_circle.hough_circle_demo1 import hough_detect_ball

if __name__ == '__main__':
    video_path_left = "D:\DIP-pythonProject\DIP_project\\videos\\left_3_406.mp4"
    video_path_right = "D:\DIP-pythonProject\DIP_project\\videos\\right_3_406.mp4"
    cap_left = cv2.VideoCapture(video_path_left)
    cap_right = cv2.VideoCapture(video_path_right)
    count = 0
    t0 = time.time()

    reconstructor = COLOR_library.map()

    tracker = TRACKING.tracking()

    # initialize the matplotlib 3D figure
    ax = plt.axes(projection='3d')
    ax.view_init(elev=90, azim=270)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    ax.set_zlim(0, 6)
    plt.ion()

    time.sleep(1)

    while cap_left.isOpened() and cap_right.isOpened():  # for 'q' key退出
        ret_left, frame_left0 = cap_left.read()
        ret_right, frame_right0 = cap_right.read()
        frame_left = frame_left0[0: 1079, 319: 1599]
        frame_right = frame_right0[0: 1079, 319: 1599]

        row, col = frame_left.shape[0], frame_left.shape[1]

        im_conca = np.concatenate((frame_left, frame_right), axis=1)
        im_conca_resize = cv2.resize(im_conca, (int(row), int(col / 2)))
        cv2.imshow("original img",im_conca_resize)

        frame_left_enlarge = cv2.copyMakeBorder(frame_left, 256, 256, 320, 320, cv2.BORDER_CONSTANT,
                                                value=[0, 0, 0])
        frame_right_enlarge = cv2.copyMakeBorder(frame_right, 256, 256, 320, 320, cv2.BORDER_CONSTANT,
                                                 value=[0, 0, 0])

        t1 = time.time()
        if count == 100:
            print("frame/s = " + str(round(100 / (t1 - t0), 3)))
            t0 = t1
            count = 0
        else:
            count = count + 1

        correct_frame_left, correct_frame_right = reconstructor.polar_correction(frame_left_enlarge, frame_right_enlarge)

        cord_left_x, cord_left_y = -1, -1
        cord_right_x, cord_right_y = -1, -1

        # cord_left_x, cord_left_y = COLOR_library.get_cord(correct_frame_left,    cord_left_x, cord_left_y)
        # cord_right_x, cord_right_y = COLOR_library.get_cord(correct_frame_right, cord_left_x, cord_left_y)


        t_left = threading.Thread(target=tracker.get_cord_2thread, args=(correct_frame_left, 0))
        t_right = threading.Thread(target=tracker.get_cord_2thread, args=(correct_frame_right, 1))
        t_left.start()
        t_right.start()
        t_left.join()
        t_right.join()

        cord_left_x, cord_left_y, cord_right_x, cord_right_y = tracker.return_cord()

        [x_3d, y_3d, z_3d] = reconstructor.get_3D_coord(cord_left_x, cord_left_y, cord_right_x, cord_right_y)

        if cord_left_x != -1 and cord_right_x != -1:
            ax.plot3D(x_3d / 1000, -1 * y_3d / 1000, z_3d / 1000, c='red', marker='.')
            plt.pause(0.0001)

        # cv2.imshow("output image", output_img)

        key = cv2.waitKey(5) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        elif key == ord("q"):
            break

    cap_left.release()
    cap_right.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

