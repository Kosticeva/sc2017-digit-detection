import cv2
import video_utils
import image_utils
import training_utils
import numpy as np

i = 0
video = cv2.VideoCapture('test samples/video-'+str(i)+'.avi')

if video.isOpened() == 0:
    print("Error!!!")

idx = 0
ann = training_utils.load_modell()
final_alphabet = training_utils.convert_output([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

while video.isOpened():

    ret, frame = video.read()
    if ret:

        img = image_utils.image_bin(image_utils.image_gray(frame))
        img_bin = image_utils.erode(image_utils.dilate(img))

        selected_regions, numbers, dimensions = video_utils.select_roi(frame.copy(), img_bin, idx, i)
        inputs = image_utils.prepare_for_ann(numbers)

        result = ann.predict(np.array(inputs, np.float32))

        f = open("results/result_"+str(i)+".txt", "a")
        f.write("\t"+str(training_utils.diss_res(result, final_alphabet)))
        f.close()

        idx = idx + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video.release()


cv2.destroyAllWindows()