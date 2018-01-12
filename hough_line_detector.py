import cv2
import image_utils
import numpy as np

ii = 0

for ii in range(0,10):
    video = cv2.VideoCapture('test samples/video-'+str(ii)+'.avi')

    if video.isOpened() == 0:
        print("Error!!!")

    idx = 0
    #ann = training_utils.load_modell()
    #final_alphabet = training_utils.convert_output([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    while video.isOpened():

        ret, frame = video.read()
        if ret:

            img = image_utils.image_bin(image_utils.image_gray(frame))
            img_bin = image_utils.erode(image_utils.dilate(img))

            lines = cv2.HoughLinesP(img_bin,1,np.pi/180,200, 0)

            a, b, c = lines.shape
            for i in range(a):
                cv2.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 255), 3,
                         cv2.LINE_AA)

            cv2.imwrite('hough_lines/video-'+str(ii)+'/frame-'+str(idx)+'.jpg', frame)
            idx = idx + 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
        print('____________________________________________________')

    video.release()
