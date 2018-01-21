from __future__ import division
import cv2
import video_utils
import image_utils
import numpy as np
import line_utils
import training_utils


#for i in range(0, 10):
for i in range(1, 10):
#for i in range(0,1):

    video = cv2.VideoCapture('test samples/video-' +str(i)+ '.avi')

    if video.isOpened() == 0:
        print("Error!!!")

    v_w = int(video.get(3))
    v_h = int(video.get(4))
    new_video = cv2.VideoWriter('test samples/video-new'+str(i)+'.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                40, (v_w, v_h))

    idx = 0
    ann = training_utils.load_modell()
    final_alphabet = training_utils.convert_output([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    blue_regions = []
    blue_dimensions = []
    blue_added = []

    green_regions = []
    green_dimensions = []
    green_added = []

    while video.isOpened():

        ret, frame = video.read()
        if ret:

            img_blue = line_utils.get_only_line(0, frame.copy())
            img_green = line_utils.get_only_line(1, frame.copy())

            lines_blue = line_utils.get_main_lines(line_utils.get_line(img_blue))
            lines_green = line_utils.get_main_lines(line_utils.get_line(img_green))

            img = image_utils.image_bin(image_utils.image_gray(frame))
            img_bin = image_utils.erode(image_utils.dilate(img))

            lines_blue_pixels = line_utils.convert_lines_to_pixels(lines_blue, img_bin)
            lines_green_pixels = line_utils.convert_lines_to_pixels(lines_green, img_bin)

            selected_regions, numbers, dimensions = video_utils.select_roi(frame.copy(), img_bin, idx, i)

            blue_regions, blue_dimensions = line_utils.check_close_ones(numbers, dimensions, lines_blue_pixels)
            green_regions, green_dimensions = line_utils.check_close_ones(numbers, dimensions, lines_green_pixels)

            f1 = open('results/close_blue' + str(i) + '.txt', "a")
            f2 = open('results/close_green' + str(i) + '.txt', "a")

            if len(blue_regions) > 0:
                blue_regions, blue_dimensions, blue_added = line_utils.check_redundancy(
                    blue_regions, blue_dimensions, blue_added, idx)

                if len(blue_regions) > 0:
                    inputs_blue = image_utils.prepare_for_ann(blue_regions)
                    result_blue = ann.predict(np.array(inputs_blue, np.float32))
                    f1.write("FRAME: " + str(idx) + "\t" + str(training_utils.diss_res(result_blue, final_alphabet)) + "\n")

            f1.close()

            if len(green_regions) > 0:
                green_regions, green_dimensions, green_added = line_utils.check_redundancy(
                    green_regions, green_dimensions, green_added, idx)

                if len(green_regions) > 0:
                    inputs_green = image_utils.prepare_for_ann(green_regions)
                    result_green = ann.predict(np.array(inputs_green, np.float32))
                    f2.write("FRAME: " + str(idx) + "\t" + str(training_utils.diss_res(result_green, final_alphabet)) + "\n")

            f2.close()

            #f = open("results/result_" + str(i) + ".txt", "a")
            # f.write("\t" + str(training_utils.diss_res(result, final_alphabet)))
            #f.close()

            for line in lines_blue:
                selected_regions = line_utils.draw_line(0, line, selected_regions)

            for line in lines_green:
                selected_regions = line_utils.draw_line(1, line, selected_regions)

            '''selected_regions = line_utils.draw_line(0, main_blue_line, selected_regions)
            selected_regions = line_utils.draw_line(1, main_green_line, selected_regions)

            cv2.rectangle(selected_regions, (main_blue_line[0], main_blue_line[1]),
                          (main_blue_line[2], main_blue_line[3]), (0, 0, 255), 1)
            cv2.rectangle(selected_regions, (main_green_line[0], main_green_line[1]),
                          (main_green_line[2], main_green_line[3]), (0, 0, 255), 1)'''

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
