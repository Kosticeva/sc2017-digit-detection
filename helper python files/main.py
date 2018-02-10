import image_utils
import training_utils
import datetime
import cv2

image_color = image_utils.load_image('training samples/training_samples.png')
img = image_utils.image_bin(image_utils.image_gray(image_color))
img_bin = image_utils.dilate_large(image_utils.erode(image_utils.dilate_large(img)))

alphabet = []
selected_regions, numbers, alphabet = image_utils.select_roi(image_color.copy(), img, alphabet)
#image_utils.display_image(selected_regions)

inputs = image_utils.prepare_for_ann(numbers)
outputs = training_utils.convert_output(alphabet)

ann = training_utils.create_ann()
ann = training_utils.train_ann(ann, inputs, outputs)
training_utils.save_model(ann)

v = datetime.datetime.now()

print(v)

#print(inputs[2])
#print(inputs[3])
#print(inputs[4])


#result = ann.predict(np.array(inputs[2:4], np.float32))
#print(result)
#print(training_utils.display_result(result, alphabet))

#v = datetime.datetime.now()

#print(v)
