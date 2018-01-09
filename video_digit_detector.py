import cv2
import video_utils
import image_utils


video = cv2.VideoCapture('test samples/demo.avi')

if video.isOpened() == 0:
    print("Error!!!")

v_w = int(video.get(3))
v_h = int(video.get(4))

print(v_w)
print(v_h)

new_video = cv2.VideoWriter('test samples/demo_gr.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, (v_w, v_h))

idx = 0
while video.isOpened():

    ret, frame = video.read()
    if ret:

        img = image_utils.image_bin(image_utils.image_gray(frame))
        img_bin = image_utils.erode(image_utils.dilate(img))

        selected_regions, numbers = image_utils.select_roi(frame.copy(), img_bin)
        cv2.imwrite("contoured_frames/frame_" + str(idx) + ".jpg", selected_regions)

        new_video.write(selected_regions)
        cv2.imshow("Video", selected_regions)

        idx = idx + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video.release()
new_video.release()

cv2.destroyAllWindows()
