# Valentin Heinitz, vheinitz@googlemail.com, 2018.01.01
# L E A R N I N G   K E R A S   WITH
# https://www.youtube.com/playlist?list=PLtPJ9lKvJ4oiz9aaL_xcZd-x0qd8G0VN_
# Using ANA-HEp2, ANCA, dDNA data sets 
#
# Find mitosis in images of Centromere-pattern cells and sort them in different
# directories

import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from scipy.misc import toimage
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from shutil import copyfile



# dimensions of the images.
img_width, img_height = 64, 64

train_dir = 'c:/tmp/ana_mit_cent/train'
val_dir = 'c:/tmp/ana_mit_cent/val'
test_dir = 'c:/tmp/ana_mit_cent/val'

nb_train_samples = 2000
nb_validation_samples = 500
nb_test_samples = 5000

epochs = 5
batch_size = 25


classes = [ 'mit', 'not_mit']
directory = 'C:/tmp/ana/cells/test/cent/'

json_file = open("ana_mit_cent.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("ana_mit_cent.h5")

loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

def get_class(prediction):
  return 1 if prediction > 0.5 else 0

right=0
wrong=0

out_mit = 'C:/tmp/ana_mit_cent/out_cent/mit'
out_not_mit = 'C:/tmp/ana_mit_cent/out_cent/not_mit'
print(directory) 
for filename in os.listdir(directory):
	fn=os.path.join(directory, filename)
	img = image.load_img(fn, target_size=(64, 64))

	x = image.img_to_array(img)
	x = x.astype('float32')
	x /= 255
	x = np.expand_dims(x, axis=0)

	prediction = loaded_model.predict(x)
	predicted = classes[get_class(prediction)]
	print(fn, prediction )
	
	if predicted == 'mit':
		copyfile(fn, os.path.join(out_mit, filename) )
	else:
		copyfile(fn, os.path.join(out_not_mit, filename))

