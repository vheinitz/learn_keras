# Valentin Heinitz, vheinitz@googlemail.com, 2018.01.01
# L E A R N I N G   K E R A S   WITH
# https://www.youtube.com/playlist?list=PLtPJ9lKvJ4oiz9aaL_xcZd-x0qd8G0VN_
# Using ANA-HEp2, ANCA, dDNA data sets 
#
# Find mitosis in images of Centromere-pattern cells and marks them

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
imgdirectory = 'C:/tmp/ana/cent/test'

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

outdir = 'C:/tmp/ana_mit_cent_real/tmpcells'

print(imgdirectory) 
for imgfilename in os.listdir(imgdirectory):

	imgfn=os.path.join(imgdirectory, imgfilename)
	print(imgfn)
	img = image.load_img(imgfn) #, target_size=(800, 600))
	
	
	objex = ["C:/Development/build/objex.exe", "--infile="+imgfn, "--outdir="+outdir, "--cleanoutdir"]
	call(objex)
	print(objex) 
	
	im = Image.open(imgfn)
	
	for cellfilename in os.listdir(outdir):
		fn=os.path.join(outdir, cellfilename)
		cellimg = image.load_img(fn , target_size=(64, 64) )
		x = image.img_to_array(cellimg)
		x = x.astype('float32')
		x /= 255
		x = np.expand_dims(x, axis=0)

		prediction = loaded_model.predict(x)
		predicted = classes[get_class(prediction)]
		
		draw = ImageDraw.Draw(im)
		if predicted == 'mit':
			parts = fn.split('_')
			x = int(parts[len(parts)-3])
			y = int(parts[len(parts)-2])
			parts =  parts[len(parts)-1].split('.')
			parts =  parts[len(parts)-2].split('x')
			w = int(parts[len(parts)-2])
			h = int(parts[len(parts)-1])
			print (x,y,w,h)
			
			draw.rectangle([(x, y),(x+w,y+h) ], outline=(0,0,255,255))
			draw.rectangle([(x+1, y+1),(x+w-1,y+h-1) ], outline=(0,0,255,255))
			draw.rectangle([(x+2, y+2),(x+w-2,y+h-2) ], outline=(0,0,255,255))
		else:
			parts = fn.split('_')
			x = int(parts[len(parts)-3])
			y = int(parts[len(parts)-2])
			parts =  parts[len(parts)-1].split('.')
			parts =  parts[len(parts)-2].split('x')
			w = int(parts[len(parts)-2])
			h = int(parts[len(parts)-1])
			print (x,y,w,h)
			draw.rectangle([(x, y),(x+w,y+h) ], outline=(255,255,255,255))
			draw.rectangle([(x+1, y+1),(x+w-1,y+h-1) ], outline=(255,255,255,255))
			draw.rectangle([(x+2, y+2),(x+w-2,y+h-2) ], outline=(255,255,255,255))

	im.save( 'C:/tmp/ana_mit_cent_real/out/'+imgfilename, "PNG" )
	im = Image.open( 'C:/tmp/ana_mit_cent_real/out/'+imgfilename )
	sys.stdout.flush()
