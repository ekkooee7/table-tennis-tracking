import time

import cv2
from DIP_project import IMAGE_PROCESS


if __name__ == '__main__':
    img = cv2.imread("img_4.png")
    img_edge = cv2.Canny(img, 50, 100)
    cv2.imshow('edge', img_edge)
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t0 = time.time()

    circles = IMAGE_PROCESS.hough_detect_ball(img, r_min=50, r_max=900, delta_r=1, num_thetas=60,
                      bin_threshold=0.5, min_edge_threshold=50, max_edge_threshold=100)
    t1 = time.time()
    print(t1-t0)

    for x, y, r, v in circles:
        img = cv2.circle(img_copy, (x, y), r, (0, 255, 0), 4)

    cv2.imshow("circle", img_copy)
    cv2.waitKey(0)



