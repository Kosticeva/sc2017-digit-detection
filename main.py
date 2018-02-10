from __future__ import division
import cv2
import video_utils
import image_utils
import numpy as np
import line_utils
import training_utils


for i in range(0, 10):

    print("--------------------------------\n"+str(i))
    video = cv2.VideoCapture('test samples/video-' + str(i) + '.avi')

    if video.isOpened() == 0:
        print("Error!!!")

    v_w = int(video.get(3))
    v_h = int(video.get(4))

    new_video = cv2.VideoWriter('noiseless_videos/video_'+str(i)+'.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, (v_w, v_h))

    idx = 0

    ann = training_utils.load_modell()
    final_alphabet = training_utils.convert_output([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    blue_regions = []
    blue_dimensions = []
    blue_added = []

    green_regions = []
    green_dimensions = []
    green_added = []

    sum_blue = 0
    sum_green = 0

    while video.isOpened(): #and (i == 1 or i == 3 or i == 4 or i == 6 or i == 9):

        ret, frame = video.read()
        if ret:

            if idx > 10:

                file1 = open('results/log_video_' + str(i) + '.txt', "a")
                file1.write('\n================== FRAME ' + str(idx) + " ==========================\n\n")
                file1.close()

                img_blue = line_utils.get_only_line(0, frame.copy())
                img_green = line_utils.get_only_line(1, frame.copy())

                lines_blue = line_utils.get_line(img_blue)
                lines_green = line_utils.get_line(img_green)

                #img_bin = image_utils.erode_large(image_utils.dilate_large(img)) #dodata spolj dilatacija
                #sa dvostruke spolj dil promenjeno na bez spolj dil

                #ovde je bilo umesto frame img_bin, da vidimo kakve cu rez dobiti sa ovim

                lines_blue_pixels = line_utils.convert_lines_to_pixels(lines_blue, image_utils.image_gray(frame))
                lines_green_pixels = line_utils.convert_lines_to_pixels(lines_green, image_utils.image_gray(frame))

                img_bin = image_utils.image_bin(image_utils.image_gray(frame), 200)
                img_bin_ed = image_utils.dilate(image_utils.erode(image_utils.dilate(img_bin, 1), 1),2)

                cv2.imwrite('noiseless_videos/video_'+str(i)+'/frame_'+str(idx)+'.jpg', img_bin_ed)

                selected_regions, numbers, dimensions = video_utils.select_roi(frame.copy(), img_bin_ed, idx, i)

                blue_regions, blue_dimensions = line_utils.check_close_ones(numbers, dimensions, lines_blue_pixels, i)
                green_regions, green_dimensions = line_utils.check_close_ones(numbers, dimensions, lines_green_pixels, i)

                #f1 = open('results/close_blue' + str(i) + '.txt', "a")
                #f2 = open('results/close_green' + str(i) + '.txt', "a")

                if len(blue_regions) > 0:
                    blue_regions, blue_dimensions, blue_added = line_utils.check_redundancy(
                        blue_regions, blue_dimensions, blue_added, idx, i)

                    if len(blue_regions) > 0:
                        inputs_blue = image_utils.prepare_for_ann(blue_regions)
                        result_blue = ann.predict(np.array(inputs_blue, np.float32))
                        to_writeB = training_utils.diss_res(result_blue, final_alphabet)
                        #f1.write("FRAME: " + str(idx) + "\t" + str(to_writeB) + "\n")
                        sum_blue = sum_blue + np.sum(to_writeB)

                        f = open('results/doc_nums_blue_'+str(i)+".txt", "a")
                        f.write('---------------------------------------------------\nFRAME\tX\tY\tW\tH\tRESULT\n')
                        for jj in range(len(result_blue)):
                            f.write(str(idx) + "\t" + str(blue_dimensions[jj][0])+"\t"+
                                    str(blue_dimensions[jj][1])+"\t"+ str(blue_dimensions[jj][2])+"\t"+
                                    str(blue_dimensions[jj][3]) + "\t" + str(to_writeB[jj])+"\n")
                        f.close()
                #f1.close()

                if len(green_regions) > 0:
                    green_regions, green_dimensions, green_added = line_utils.check_redundancy(
                        green_regions, green_dimensions, green_added, idx, i)

                    if len(green_regions) > 0:
                        inputs_green = image_utils.prepare_for_ann(green_regions)
                        result_green = ann.predict(np.array(inputs_green, np.float32))
                        to_writeG = training_utils.diss_res(result_green, final_alphabet)
                        #f2.write("FRAME: " + str(idx) + "\t" + str(to_writeG) + "\n")
                        sum_green = sum_green + np.sum(to_writeG)

                        f = open('results/doc_nums_green_' + str(i) + ".txt", "a")
                        f.write('---------------------------------------------------\nFRAME\tX\tY\tW\tH\tRESULT\n')
                        for jj in range(len(result_green)):
                            f.write(str(idx) + "\t" + str(green_dimensions[jj][0]) + "\t" +
                                    str(green_dimensions[jj][1]) + "\t" + str(green_dimensions[jj][2]) + "\t" +
                                    str(green_dimensions[jj][3]) + "\t" + str(to_writeG[jj]) + "\n")
                        f.close()

               # f2.close()

                '''f = open("results/result_" + str(i) + ".txt", "a")
                f.write("\t" + str(training_utils.diss_res(result, final_alphabet)))
                f.close()'''

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

                cv2.imwrite('contoured_frames/video-'+str(i)+'/frame_'+str(idx)+'.jpg', selected_regions)
                new_video.write(selected_regions)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print(idx)
            idx = idx + 1
        else:
            break

    f = open("results/out.txt", "a")
    f.write("video-"+str(i)+".avi\t"+str(sum_blue-sum_green)+"\n")
    f.close()

    video.release()
    new_video.release()