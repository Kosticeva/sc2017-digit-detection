import image_utils
import cv2
import numpy as np


def get_only_line(mode, image):

    if mode == 0:
        ret, image_bin = cv2.threshold(image[:, :, mode], 200, 255, cv2.THRESH_BINARY)
        img_bin = image_utils.dilate(image_utils.erode(image_utils.dilate(image_utils.erode(image_bin, 3), 3), 3), 3)
    elif mode == 1:
        img_bin = image_utils.erode(image_utils.dilate(image_utils.erode(image[:, :, mode], 2), 2), 2)
    else:
        img_bin = []

    return img_bin


def get_line(img):

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 200, 10)
    return lines


def get_main_lines(lines):

    main_tuples = []

    for j in range(len(lines)):
        for x1, y1, x2, y2 in lines[j]:
            d = np.sqrt(np.power(x1-x2, 2)+np.power(y1-y2, 2))
            if d > 0:
                main_tuples.append((x1, y1, x2, y2))

    return main_tuples


def draw_line(mode, line, img):

    color = (0, 0, 0)
    if mode == 0:
        color = (0, 255, 255)
    else:
        color = (0, 0, 255)

    cv2.line(img, (line[0][0], line[0][1] + 4), (line[0][2], line[0][3] + 4), color, 1, cv2.LINE_AA)

    return img


#regions = svi regioni na slici, dimensions = koord ogd regiona, coordinates = koordinate pikslea linije
def check_close_ones(regions, dimensions, coordinates, video):

    #kada se kontura nekim delom nadje u prostoru koji zauzimaju koordinate
    #ulazi u new_regions
    file1 = open('results/log_video_' + str(video) + '.txt', "a")
    file1.write('\n------------------- PROXIMITY CHECK -------------------------\n')

    new_regions = []
    new_dimensions = []
    idx = 0
    for x, y, w, h in dimensions:
        if check_if_matches_line((x, y, x+w, y+h), coordinates):
            new_regions.append(regions[idx])
            new_dimensions.append(dimensions[idx])
            file1.write('### I AM CLOSE TO LINE (dimensions: '+str(dimensions[idx])+')\n')
        else:
            file1.write('@@@ I AM TOO FAR (dimensions: ' + str(dimensions[idx]) + ')\n')

        idx = idx + 1

    file1.close()
    return new_regions, new_dimensions


#contour = x1,y1,x2,y2
#proverava da li se gornji levi ugao konture
#poklapa sa nekim pikselom linije
def check_if_matches_line(contour, pixel_lines):

    for line in pixel_lines:
        for pixel in line:
            if contour[0] == pixel[0] and contour[1] == pixel[1]:
                return True
            elif contour[0] == pixel[0] and (contour[1] - pixel[1]) in range(0, 4):
                return True

    return False


def convert_lines_to_pixels(lines, image):

    lines_pixels = []
    a, b, c = lines.shape

    for i in range(a):
        lines_pixels.append(create_line_iterator(np.array([lines[i][0][0], lines[i][0][1]]),
                                                 np.array([lines[i][0][2], lines[i][0][3]]),
                                                 image))

    lines_pixels = move_down_lines(lines_pixels)

    return lines_pixels


"""
Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

Parameters:
    -P1: a numpy array that consists of the coordinate of the first point (x,y)
    -P2: a numpy array that consists of the coordinate of the second point (x,y)
    -img: the image being processed

Returns:
    -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])     
"""


def create_line_iterator(P1, P2, img):
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype = np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X: #vertical line segment
       itbuffer[:, 0] = P1X
       if negY:
           itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
       else:
           itbuffer[:, 1] = np.arange(P1Y+1, P1Y+dYa+1)
    elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:, 0] = np.arange(P1X-1, P1X-dXa-1, -1)
       else:
           itbuffer[:, 0] = np.arange(P1X+1, P1X+dXa+1)
    else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:, 1] = np.arange(P1Y-1, P1Y-dYa-1, -1)
           else:
               itbuffer[:, 1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:, 0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:, 0] = np.arange(P1X-1,P1X-dXa-1, -1)
           else:
               itbuffer[:, 0] = np.arange(P1X+1, P1X+dXa+1)
           itbuffer[:, 1] = (slope*(itbuffer[:, 0]-P1X)).astype(np.int) + P1Y

    #Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX<imageW) & (colY < imageH)]

    #Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


def move_down_lines(line_pixels):

    for line in line_pixels:
        for pixel in line:
            pixel[1] = pixel[1] + 6

    return line_pixels


def check_redundancy(regions, dimensions, added, idx, video):

    file1 = open('results/log_video_'+str(video)+'.txt', "a")
    file1.write('\n-------------------- REDUNDANCY CHECK --------------------------\n')

    new_regions = []
    new_dimensions = []
    i = 0
    for region in regions:
        flag_idx = check_if_added(dimensions[i], added, idx)
        if flag_idx < 0:
            file1.write('*** FOUND NEW REGION (dimensions: '+str(dimensions[i])+')\n')
            new_regions.append(region)
            new_dimensions.append(dimensions[i])
            added.append((dimensions[i][0], dimensions[i][1], dimensions[i][2], dimensions[i][3], idx))
        else:
            file1.write('+++ REGION APPEARING AGAIN (dimensions: ' + str(dimensions[i]) + ')\n')
            added.remove(added[flag_idx])
            added.append((dimensions[i][0], dimensions[i][1], dimensions[i][2], dimensions[i][3], idx))

        i = i + 1

    file1.close()
    return new_regions, new_dimensions, added


def check_if_added(checker, added, idx):

    i = 0
    for region in added:
        '''if ((abs(region[2] - checker[2]) < 2 and region[3] == checker[3]) \
                or (region[2] == checker[2] and abs(region[3] - checker[3]) < 2)
                or ()) and abs(idx - region[4]) < 15\
                and (abs(checker[0] - region[0]) < 4 and abs(checker[1] - region[1]) < 4):'''
        if abs(idx - region[4]) < 15\
                and (abs(checker[0] - region[0]) < 6 and abs(checker[1] - region[1]) < 6):
            return i
        i = i + 1

    return -1

