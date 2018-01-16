import cv2
import video_utils
import image_utils
import training_utils
import numpy as np
import line_utils

#for i in range(0, 10):

i = 0

video = cv2.VideoCapture('test samples/video-'+str(i)+'.avi')

if video.isOpened() == 0:
    print("Error!!!")

v_w = int(video.get(3))
v_h = int(video.get(4))
new_video = cv2.VideoWriter('test samples/video-new'+str(i)+'.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, (v_w, v_h))

idx = 0
ann = training_utils.load_modell()
final_alphabet = training_utils.convert_output([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

while video.isOpened():

    ret, frame = video.read()
    if ret:

        img_blue = line_utils.get_only_line(0, frame.copy())
        img_green = line_utils.get_only_line(1, frame.copy())

        if idx == 0:
            cv2.imwrite('results/img_blue'+str(i)+".jpg", img_blue)
            cv2.imwrite('results/img_green' + str(i) + ".jpg", img_green)

        line_blue = line_utils.get_line(img_blue)
        line_green = line_utils.get_line(img_green)

        main_blue_line = line_utils.get_main_line(line_blue)
        main_green_line = line_utils.get_main_line(line_green)

        img = image_utils.image_bin(image_utils.image_gray(frame))
        img_bin = image_utils.erode(image_utils.dilate(img))

        selected_regions, numbers, dimensions = video_utils.select_roi(frame.copy(), img_bin, idx, i)

        inputs = image_utils.prepare_for_ann(numbers)
        result = ann.predict(np.array(inputs, np.float32))

        f = open("results/result_" + str(i) + ".txt", "a")
        f.write("\t" + str(training_utils.diss_res(result, final_alphabet)))
        f.close()

        selected_regions = line_utils.draw_line(0, main_blue_line, selected_regions)
        selected_regions = line_utils.draw_line(1, main_green_line, selected_regions)

        new_video.write(selected_regions)
        idx = idx + 1
        print(idx)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video.release()
new_video.release()


