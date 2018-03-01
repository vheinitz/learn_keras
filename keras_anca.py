from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense
import keras



# dimensions of our images.
img_width, img_height = 64, 64
train_dir = 'c:/tmp/anca/train'
val_dir = 'c:/tmp/anca/val'
test_dir = 'c:/tmp/anca/test'
nb_train_samples = 10000
nb_val_samples = 1000
nb_test_samples = 1000
epochs = 10
batch_size = 25

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

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
model.add(Dense(3))
model.add(Activation('sigmoid'))
	
# Компилируем модель
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Создаем генератор данных для обучения
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
	rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=True,  # randomly flip images
	vertical_flip=True
    )

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
	rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=True,  # randomly flip images
	vertical_flip=True
    )
	
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
			    rescale=1. / 255
	)
	
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical')

# Создаем генератор данных для валидации
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical')
	

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.summary()
	
# Обучаем модель с помощью генератора
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples  // batch_size / 10,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = nb_val_samples  // batch_size / 10)

print("Saving net ..")
# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("anca.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("anca.h5")
print("Finished saving")

score = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

"""	
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

	base_model = applications.VGG16(weights='imagenet', include_top=False)
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
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
	rotation_range=90,  # randomly rotate images in the range (degrees, 0 to 180)
	width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
	height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
	horizontal_flip=True,  # randomly flip images
	vertical_flip=True
    )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
			    rescale=1. / 255
	)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')
	
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size*10,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_val_samples // batch_size)

	
#score = model.evaluate(validation_generator, verbose=1)
score = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
	
model.save_weights('first_try.h5')
"""