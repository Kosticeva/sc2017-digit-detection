import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs, min_thresh = 70):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, min_thresh, 255, cv2.THRESH_BINARY)
    return image_bin


def invert(image):
    return 255-image


def dilate(image, i):
    kernel = np.ones((i,i)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode(image, i):
    kernel = np.ones((i,i)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def dilate_large(image):
    kernel = np.ones((3, 3))  # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)


def erode_large(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)


def select_roi(image_orig, image_binn, alphabet):

    #cv2.imwrite("C:/Users/Jelena/Desktop/regions_preRect.png", image_binn)
    img, contours, hierarchy = cv2.findContours(image_binn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []

    #filee = open("C:/Users/Jelena/Desktop/regions.txt", "a")
    #filee.write("INDEX\tX_COORD\tY_COORD\tWIDTH\tHEIGHT\tAREA\tFRAME")

    idx = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)

        #network training trenchold
        if area > 0 and area < 150 and ((h > 10 and w < 10) or (h > 7 and w > 7)):
            region = image_binn[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255,0,0), 1)

            #filee.write("\n"+str(idx)+"\t"+str(x)+"\t"+str(y)+"\t"+str(w)+"\t"+str(h)+"\t"+str(area))
            idx = idx + 1

            out = y//100
            alphabet.append(out)

        #filee.write("\n"+str(idx)+"\t"+str(x)+"\t"+str(y)+"\t"+str(w)+"\t"+str(h)+"\t"+str(area))
        idx = idx + 1

    filee.close()
    regions_array = [region[0] for region in regions_array]
    return image_orig, regions_array, alphabet


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1

    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            if image[i][j] > 127:
                image[i][j] = 1
            else:
                image[i][j] = 0

    return image


def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scaled = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scaled))

    return ready_for_ann


def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

    cv2.imwrite("results/regions.png", image)


def do_er_dil(image, i, j):
    image = erode(image, i)
    return dilate(image, j)