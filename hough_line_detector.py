import cv2
import line_utils
import numpy as np

ii = 0

for ii in range(0,10):
#for ii in range(3,4):
    video = cv2.VideoCapture('test samples/video-'+str(ii)+'.avi')

    if video.isOpened() == 0:
        print("Error!!!")

    idx = 0
    v_w = int(video.get(3))
    v_h = int(video.get(4))
    #ann = training_utils.load_modell()
    #final_alphabet = training_utils.convert_output([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    new_video1 = cv2.VideoWriter('hough_lines/video-' + str(ii) + '/blue.avi',
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, (v_w, v_h))
    new_video2 = cv2.VideoWriter('hough_lines/video-' + str(ii) + '/green.avi',
                                 cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, (v_w, v_h))

    while video.isOpened():

        ret, frame = video.read()
        if ret:

            img_blue = line_utils.get_only_line(0, frame.copy())
            img_green = line_utils.get_only_line(1, frame.copy())

            #lines_blue = line_utils.get_main_lines(line_utils.get_line(img_blue))
            #lines_green = line_utils.get_main_lines(line_utils.get_line(img_green))

            lines_blue = line_utils.get_line(img_blue)
            lines_green = line_utils.get_line(img_green)

            frame1 = frame.copy()
            frame2 = frame.copy()

            f = open('hough_lines/video-' + str(ii) + '/blue.txt', 'a')
            f.write("=============FRAME "+str(idx)+"====================\nX1\tY1\tX2\tY2\n")
            for line in lines_blue:
                frame1 = line_utils.draw_line(0, line, frame1)
                f.write(str(line[0][0])+'\t'+str(line[0][1])+'\t'+str(line[0][2])+'\t'+str(line[0][3])+'\n')
            f.close()

            f = open('hough_lines/video-' + str(ii) + '/green.txt', 'a')
            f.write("=============FRAME " + str(idx) + "====================\nX1\tY1\tX2\tY2\n")
            for line in lines_green:
                frame2 = line_utils.draw_line(1, line, frame2)
                f.write(str(line[0][0]) + '\t' + str(line[0][1]) + '\t' + str(line[0][2]) + '\t' + str(line[0][3]) + '\n')
            f.close()

            new_video1.write(frame1)
            new_video2.write(frame2)

            print(idx)
            idx = idx + 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    video.release()
    new_video1.release()
    new_video2.release()

