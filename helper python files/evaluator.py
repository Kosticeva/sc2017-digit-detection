import image_utils
import training_utils
import numpy as np

image_color = image_utils.load_image('test samples/test_samples(0,25).png')
img = image_utils.image_bin(image_utils.image_gray(image_color))
img_bin = image_utils.erode(image_utils.dilate(img))

alphabet = []
selected_regions, numbers, alphabet = image_utils.select_roi(image_color.copy(), img, alphabet)
image_utils.display_image(selected_regions)

print(alphabet)

inputs = image_utils.prepare_for_ann(numbers)
outputs = training_utils.convert_output(alphabet)

ann = training_utils.load_modell()

#print(outputs)

result = ann.predict(np.array(inputs, np.float32))

final_alphabet = training_utils.convert_output([0,1,2,3,4,5,6,7,8,9])

print(result)
print(training_utils.display_result(result, final_alphabet, alphabet))

