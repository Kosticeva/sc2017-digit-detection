from __future__ import division
import cv2
import video_utils
import image_utils
import training_utils
import numpy as np
import line_utils


#i = 0
for i in range(0, 10):

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

            k_blue_line, n_blue_line = line_utils.get_line_coef(main_blue_line)
            k_green_line, n_green_line = line_utils.get_line_coef(main_green_line)

            img = image_utils.image_bin(image_utils.image_gray(frame))
            img_bin = image_utils.erode(image_utils.dilate(img))

            selected_regions, numbers, dimensions = video_utils.select_roi(frame.copy(), img_bin, idx, i)

            blue_close_ones = line_utils.check_close_ones(k_blue_line, n_blue_line, main_blue_line, dimensions, numbers)
            green_close_ones = line_utils.check_close_ones(k_green_line, n_green_line, main_green_line, dimensions, numbers)

            #inputs = image_utils.prepare_for_ann(numbers)
            #result = ann.predict(np.array(inputs, np.float32))

            if len(blue_close_ones) > 0:
                inputs_blue = image_utils.prepare_for_ann(blue_close_ones)
                result_blue = ann.predict(np.array(inputs_blue, np.float32))
                f_blue = open('results/close_blue' + str(i) + '.txt', "a")
                f_blue.write("\nFRAME: " + str(idx) + "\t" + str(training_utils.diss_res(result_blue, final_alphabet)))
                f_blue.close()

            if len(green_close_ones) > 0:
                inputs_green = image_utils.prepare_for_ann(green_close_ones)
                result_green = ann.predict(np.array(inputs_green, np.float32))
                f_green = open('results/close_green' + str(i) + '.txt', "a")
                f_green.write("\nFRAME: " + str(idx) + "\t" + str(training_utils.diss_res(result_green, final_alphabet)))
                f_green.close()

            #f = open("results/result_" + str(i) + ".txt", "a")
            # f.write("\t" + str(training_utils.diss_res(result, final_alphabet)))
            #f.close()

            selected_regions = line_utils.draw_line(0, main_blue_line, selected_regions)
            selected_regions = line_utils.draw_line(1, main_green_line, selected_regions)

            cv2.imwrite('contoured_frames/video_'+str(i)+'/frame_'+str(idx)+'.jpg', selected_regions)

            new_video.write(selected_regions)
            idx = idx + 1
            print(idx)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    video.release()
    new_video.release()


