import cv2
import pyzed.sl as sl
import numpy as np

# 乒乓球位置识别，加入了指示移动方向的箭头

def empty(a):
    pass


def get_table(img):
    # initiate the basic mask of the selected area
    mask = np.zeros(img.shape, dtype=np.uint8)
    # get the whole white color mask for further mix with mask
    channels = img.shape[2]
    ignore_mask_color = (255,) * channels
    a = 370
    b = 217 - 70
    c = 961
    d = 214 - 70
    e = 1272
    f = 476
    g = 118
    h = 470

    image_segment = sub_get_table(img, mask, ignore_mask_color, a, b,
                                  c, d, e, f, g, h)

    return image_segment


def sub_get_table(img, mask, ignore_mask_color, a, b, c, d, e, f, g, h):
    # the coordinate of the selected area without the white counter of the table
    # roi_corners = np.array([[(422, 315), (908, 315), (1017, 603), (338, 595)]], dtype=np.int32)
    # whole white counter included.
    roi_corners = np.array([[(a, b), (c, d), (e, f), (g, h)]],
                           dtype=np.int32)
    # 创建mask层
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # 为每个像素进行与操作，除mask区域外，全为0
    masked_image = cv2.bitwise_and(img, mask)
    # anything changed here should pay attention to the code where the mark A locates
    return masked_image[d:f, g:e]


def draw_direction(img, lx, ly, nx, ny):
    # 根据上一位置与当前位置计算移动方向并绘制箭头
    dx = nx - lx
    dy = ny - ly
    if abs(dx) < 4 and abs(dy) < 4:
        dx = 0
        dy = 0
    else:
        r = (dx ** 2 + dy ** 2) ** 0.5
        dx = int(dx / r * 40)
        dy = int(dy / r * 40)
        # print(dx, dy)
    cv2.arrowedLine(img, (60, 100), (60 + dx, 100 + dy), (0, 255, 0), 2)
    # print(nx-lx, ny-ly)   # 噪声一般为+-1
    # cv2.arrowedLine(img, (150, 150), (150+(nx-lx)*4, 150+(ny-ly)*4), (0, 0, 255), 2, 0, 0, 0.2)


class tracking:
    def __init__(self):
        self.WIDTH = 720       # width为垂直方向
        self.HEIGHT = 2560       # height为水平方向

    def runing(self):
        init = sl.InitParameters()

        cam = sl.Camera()

        if not cam.is_opened():
            print("Opening ZED Camera...")
        status = cam.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()
        cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 20)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 99)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 9)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 9)

        runtime = sl.RuntimeParameters()
        mat = sl.Mat()

        while True:
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.SIDE_BY_SIDE)
                # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                # cv2.imshow("image", mat.get_data())

                a = 1
                image = mat.get_data() * a

                self.WIDTH = image.shape[0]
                self.HEIGHT = image.shape[1]

                print(self.WIDTH,self.HEIGHT)

                image_resize = cv2.resize(image, [ int(self.HEIGHT * 0.5), int(self.WIDTH * 0.5)])
                cv2.imshow('input image', image_resize)

                key = cv2.waitKey(5) & 0xff
                if key == ord(" "):
                    cv2.waitKey(0)
                elif key == ord("q"):
                    break

        cam.close()
        cv2.destroyAllWindows()




if __name__ == '__main__':
    track = tracking()
    track.runing()

