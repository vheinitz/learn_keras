from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model


# dimensions of our images.
img_width, img_height = 64, 64
train_dir = 'c:/tmp/anca/train'
val_dir = 'c:/tmp/anca/val'
test_dir = 'c:/tmp/anca/test'
nb_train_samples = 10000
nb_val_samples = 1000
nb_test_samples = 1000
epochs = 20
batch_size = 25

# Загружаем сеть VGG16 без части, которая отвечает за классификацию
base_model = applications.VGG16(weights='imagenet', include_top=False)

# Добавляем слои классификации
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# Выходной слой с двумя нейронами для классов "x" и "c"
predictions = Dense(3, activation='softmax')(x)

# Составляем сеть из двух частей
model = Model(inputs=base_model.input, outputs=predictions)

# "Замораживаем" сверточные уровни сети VGG16
# Обучаем только вновь добавленные слои
for layer in base_model.layers:
    layer.trainable = False
	
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
	
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
			    rescale=1. / 255
	)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.summary()
	
# Обучаем модель с помощью генератора
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // 100,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps = nb_val_samples // 20)

print("Saving net ..")
# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("anca_apc.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("anca_apc.h5")
print("Finished saving")

score = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
