import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from scipy.misc import toimage
import os


# Список классов
classes = ['aanca', 'panca', 'canca']
directory = 'C:/tmp/anca/test/'

json_file = open("anca.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("anca.h5")

loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

for c in classes:
	aktdir=directory+c
	print(aktdir)
	for filename in os.listdir(aktdir):
		fn=os.path.join(aktdir, filename)
		#print(fn)
		img = image.load_img(fn, target_size=(64, 64))
		#plt.imshow(img)
		#plt.show()

		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		prediction = loaded_model.predict(x)

		#print(prediction)
		print(c + " : " + classes[np.argmax(prediction)], " : ", prediction)

