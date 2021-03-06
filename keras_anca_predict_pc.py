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

train_dir = 'c:/tmp/anca_pc/train'
val_dir = 'c:/tmp/anca_pc/val'
test_dir = 'c:/tmp/anca_pc/test'

nb_train_samples = 2000
nb_validation_samples = 500
nb_test_samples = 500

epochs = 5
batch_size = 25


classes = [ 'canca', 'panca']
directory = 'C:/tmp/anca_pc/test/'

json_file = open("anca_pc.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("anca_pc.h5")

loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

def get_class(prediction):
  return 1 if prediction > 0.5 else 0

for c in classes:
	aktdir=directory+c
	print(aktdir)
	cnt=0
	for filename in os.listdir(aktdir):
		cnt = cnt+1
		if cnt > 100:
			break
		fn=os.path.join(aktdir, filename)
		#print(fn)
		img = image.load_img(fn, target_size=(64, 64))
		#plt.imshow(img)
		#plt.show()

		x = image.img_to_array(img)
		x = x.astype('float32')
		x /= 255
		x = np.expand_dims(x, axis=0)
		#x = preprocess_input(x)

		prediction = loaded_model.predict(x)

		#print(prediction)
		print(c + " : " + classes[get_class(prediction)], " : ", prediction)

datagen = ImageDataGenerator( rescale=1. / 255 )	
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
	
prediction = loaded_model.predict_generator(test_generator)
	
score = loaded_model.evaluate_generator(test_generator, 500 )

print('Test loss:', score[0])
print('Test accuracy:', score[1])