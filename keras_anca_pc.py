"""
$ for d in test train val; do for c in canca panca aanca; do for f in $(ls -1 $d/$c); do /C/Program\ Files/ImageMagick-7.0.7-Q16/magick.exe $d/$c/$f -channel G -resize 64x64 $d/$c/$f; done; done; done

"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of the images.
img_width, img_height = 64, 64

train_dir = 'c:/tmp/ndna/train'
val_dir = 'c:/tmp/ndna/val'
test_dir = 'c:/tmp/ndna/test'
out_dir = 'c:/tmp/ndna/out'

nb_train_samples = 10000
nb_validation_samples = 1000
nb_test_samples = 500

epochs = 20
batch_size = 25

#Is right got gray-scale images? Shouldn't be 1 instead of 3
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#Define model as proposed in keras tutorials
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

test_datagen = ImageDataGenerator( rescale=1. / 255 )
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
	rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=True,  # randomly flip images
	vertical_flip=True
    )

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
	
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
	save_to_dir=out_dir)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#Save model architecture
model_json = model.to_json()
json_file = open("ndna.json", "w")
json_file.write(model_json)
json_file.close()

#Save model weights
model.save_weights("ndna.h5")
print("Finished saving")

score = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

score = model.predict_generator(train_generator, nb_test_samples // batch_size)

#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#print('Score', score)
