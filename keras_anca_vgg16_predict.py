import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from scipy.misc import toimage
import os


# Список классов
classes = ['panca', 'canca', 'aanca']
directory = 'C:/tmp/anca/test/canca'

json_file = open("vgg16_anca.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("vgg16_anca.h5")

loaded_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

for filename in os.listdir(directory):
	fn=os.path.join(directory, filename)
	print(fn)
	img = image.load_img(fn, target_size=(64, 64))
	#plt.imshow(img)
	#plt.show()

	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	prediction = loaded_model.predict(x)

	print(prediction)
	print(classes[np.argmax(prediction)])

