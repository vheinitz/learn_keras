# Valentin Heinitz, vheinitz@googlemail.com, 2018.01.01
# L E A R N I N G   K E R A S   WITH
# https://www.youtube.com/playlist?list=PLtPJ9lKvJ4oiz9aaL_xcZd-x0qd8G0VN_
# Using ANA-HEp2, ANCA, dDNA data sets 
#
# Find in real images cells for 7 patterns trained with tiles, not single cells

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
from subprocess import call
import sys
import matplotlib.patches as patches
from PIL import Image, ImageDraw


# dimensions of the images.
img_width, img_height = 128, 128

train_dir = 'c:/tmp/ana/rects/train'
val_dir = 'c:/tmp/ana/rects/val'
test_dir = 'c:/tmp/ana/rects/test'

nb_train_samples = 2000
nb_validation_samples = 500
nb_test_samples = 5000

epochs = 5
batch_size = 25


classes=['cent' ,  'cyt' , 'env', 'hom', 'nuc', 'nucdot', 'spe' ]
#classes=['cyt' , 'hom',  'spe' ]
directory = 'C:/tmp/ana/rects/test/'
imgdirectory = 'C:/tmp/ana/nuc/test'

json_file = open("ana_rects.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("ana_rects.h5")

loaded_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


outdir = 'C:/tmp/ana_mit_cent_real/tmpcells'

print(imgdirectory) 
for imgfilename in os.listdir(imgdirectory):

	imgfn=os.path.join(imgdirectory, imgfilename)
	#print(imgfn)
	img = image.load_img(imgfn) #, target_size=(800, 600))
	
	
	objex = ["C:/Development/build/objex.exe", "--infile="+imgfn, "--outdir="+outdir, "--cleanoutdir", '--rects=300x300']
	call(objex)
	#print(objex) 
	
	im = Image.open(imgfn)
	r={}
	r['cyt'] = 0
	r['hom'] = 0
	r['spe'] = 0
	r['env'] = 0
	r['nuc'] = 0
	r['nucdot'] = 0
	r['cent'] = 0
	
	for cellfilename in os.listdir(outdir):
		fn=os.path.join(outdir, cellfilename)
		cellimg = image.load_img(fn , target_size=(img_width, img_height) )
		x = image.img_to_array(cellimg)
		x = x.astype('float32')
		x /= 255
		x = np.expand_dims(x, axis=0)

		prediction = loaded_model.predict(x)
		pc = classes[np.argmax(prediction)]
		#print( prediction, pc )
		r[pc] = r[pc]+1
		#print(r);

	pc = classes[np.argmax(r)]
	print( imgfn, r)
	sys.stdout.flush()
	