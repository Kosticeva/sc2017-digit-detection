import cv2
import image_utils


def select_roi(image_orig, image_binn, idz, video_num):

    img, contours, hierarchy = cv2.findContours(image_binn.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []

    #filee = open("results/regions"+str(video_num)+".txt", "a")
    filee = open("results/result_" + str(video_num) + ".txt", "a")
    filee.write("\n\nINDEX\tX_COORD\tY_COORD\tWIDTH\tHEIGHT\tAREA\tFRAME")

    idx = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)

        if area > 7 and area < 300 and (h > 5 and ((h > 7 and w<=5) or (w>5 and area>15))) and (h<30 and w < 30) :
            region = image_binn[y:y + h + 1, x:x + w + 1]
            regions_array.append([image_utils.resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255,0,0), 1)

            filee.write("\n"+str(idx)+"\t"+str(x)+"\t"+str(y)+"\t"+str(w)+"\t"+str(h)+"\t"+str(area)+"\t"+str(idz))
            idx = idx + 1

    filee.close()
    regions_array = [region[0] for region in regions_array]
    return image_orig, regions_array
