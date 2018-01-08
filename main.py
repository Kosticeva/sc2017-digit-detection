import image_utils
import training_utils
import numpy as np
import datetime

#TODO
#Smanjiti dataset - povecati parametre praga odabira regiona od interesa
#Smanjiti epohe
#Odraditi serijalizaciju neuronske mreze - .pkl fajl (POTRAZI NET)
#Jbg ostavi da radi celu noc pa ces videti sta da radis
#Ubaci ispis pored rezultata - kada je zavrseno obucavanje mreze i kada je zavrsena evaluacija

image_color = image_utils.load_image('C:/Users/Jelena/Desktop/Fakultet/7. semestar/Soft/test_samples(0,75).png')
img = image_utils.image_bin(image_utils.image_gray(image_color))
img_bin = image_utils.erode(image_utils.dilate(img))

alphabet = []
selected_regions, numbers, alphabet = image_utils.select_roi(image_color.copy(), img, alphabet)
image_utils.display_image(selected_regions)

inputs = image_utils.prepare_for_ann(numbers)
outputs = training_utils.convert_output(alphabet)

ann = training_utils.create_ann()
ann = training_utils.train_ann(ann, inputs, outputs)
training_utils.save_model(ann)

v = datetime.datetime.now()

print(v)

print(inputs[2])
print(inputs[3])
print(inputs[4])


result = ann.predict(np.array(inputs[2:4], np.float32))
print(result)
print(training_utils.display_result(result, alphabet))

v = datetime.datetime.now()

print(v)
