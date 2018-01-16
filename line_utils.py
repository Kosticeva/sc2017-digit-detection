import image_utils
import cv2
import numpy as np


def get_only_line(mode, image):

    if mode < 2:
        img_bin = image_utils.dilate(image_utils.erode(image_utils.erode(image_utils.erode(image_utils.dilate(image[:,:,mode])))))
    else:
        img_bin = []

    return img_bin


def get_line(img):

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 200, 0)
    return lines


def get_main_line(lines):

    max_d = 0

    main_tuple = []

    for j in range(len(lines)):
        for x1,y1,x2,y2 in lines[j]:
            d = np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
            if d > max_d:
                main_tuple = (x1,y1,x2,y2)
                max_d = d

    return main_tuple


def draw_line(mode, line, img):

    color = (0,0,0)
    if mode == 0:
        color = (0,255,255)
    else:
        color = (0, 0, 255)

    cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, 1, cv2.LINE_AA)

    return img
