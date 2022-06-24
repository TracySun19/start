import cv2
import numpy as np


def pic_read(path):
    img = cv2.imread(path)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    pass


if __name__ == '__main__':

    img_path = './plane.png'
    pic_read(img_path)