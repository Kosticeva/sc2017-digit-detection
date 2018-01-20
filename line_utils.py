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


def get_line_coef(points):
    k = (points[1] - points[3]) / (points[0] - points[2])
    n = -k * points[0] + points[1]

    return round(k), round(n)


def check_close_ones(k,n,line, dimensions, regions):

    #posto kontura nestaje u trenutku kada pipne liniju
    #treba da proverimo razlike izmedju konture i linije i to:
    #1) po uglovima konture - obe tacke
    #i to za sve tacke na liniji line
    dots = get_dots_from_line(k,n,line)

    close_ones = []
    #samo po

    idx = 0
    for x,y,w,h in dimensions:
        for dot in dots:
            if (abs(dot[0]-x)<3 and abs(dot[1]-y)<3) or (abs(dot[0]-(x+w))<3 and abs(dot[1]-(y+h))<3):
                close_ones.append(regions[idx])
                break
        idx = idx + 1

    return close_ones


def get_dots_from_line(k, n, line):

    x1 = line[0]
    x2 = line[2]

    dots = []

    for x in range(x1,x2+1):
        dots.append([x, k*x+n])

    return dots
