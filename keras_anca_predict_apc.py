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


# dimensions of the images.
img_width, img_height = 64, 64

train_dir = 'c:/tmp/anca/train'
val_dir = 'c:/tmp/anca/val'
test_dir = 'c:/tmp/anca/test'

nb_train_samples = 10000
nb_validation_samples = 500
nb_test_samples = 500

epochs = 5
batch_size = 100


classes = [ 'aanca', 'canca', 'panca' ]
directory = 'C:/tmp/anca/test/'

directory = 'C:/tmp/cells/xxx'

json_file = open("anca_apc.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("anca_apc.h5")

loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

right=0
wrong=0

for c in classes:
	aktdir=directory+c
	print(aktdir)
	cnt=0
	for filename in os.listdir(aktdir):
		cnt = cnt+1
		#if cnt > 100:
		#	break
		fn=os.path.join(aktdir, filename)
		#print(fn)
		img = image.load_img(fn, target_size=(64, 64))
		#plt.imshow(img)
		#plt.show()

		x = image.img_to_array(img)
		x = x.astype('float32')
		x /= 255
		x = np.expand_dims(x, axis=0)
		

		prediction = loaded_model.predict(x)

		if c == classes[np.argmax(prediction)]:
			right = right +1
		else:
			wrong = wrong +1

		print(c + " : " + classes[np.argmax(prediction)], " : ", right, wrong)
"""
datagen = ImageDataGenerator( rescale=1. / 255 )	
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
	
prediction = loaded_model.predict_generator(test_generator)
	
score = loaded_model.evaluate_generator(test_generator, 500 )

print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""