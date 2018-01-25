import cv2
import image_utils

for i in range(0, 10):
    video = cv2.VideoCapture('test samples/video-' + str(i) + '.avi')

    if video.isOpened() == 0:
        print("Error!!!")

    idx = 0
    while video.isOpened():

        ret, frame = video.read()

        if ret:

            removed_noise_frame = image_utils.image_bin(image_utils.image_gray(frame), 200)
            removed_noise_frame = image_utils.erode(image_utils.dilate(removed_noise_frame))

            cv2.imwrite('noiseless_videos/video_' + str(i) + '/frame_' + str(idx) + '.jpg', removed_noise_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

        print(idx)
        idx = idx + 1

    video.release()