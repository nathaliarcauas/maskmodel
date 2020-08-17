from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'Split/train',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical',
        color_mode = 'rgb')
validation_generator = test_datagen.flow_from_directory(
        'Split/test',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical',
        color_mode = 'rgb')

print("train", list(train_generator.class_indices.items()))
print("val", list(validation_generator.class_indices.items()))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(200, 200,3), activation = 'relu'))  
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25)) 

model.add(Conv2D(64, (3, 3), padding='same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dense(512, activation ='relu'))
model.add(Dropout(0.25))
model.add(Dense(512, activation ='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation ='softmax')) 

opt = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")

# print(model.summary())

# path = "weights/"
# mcp_save = ModelCheckpoint(path + 'new_weights_epoch_{epoch:03d}_loss_{val_loss:.2f}.hdf5', save_best_only=True, monitor='val_loss', mode='min')


# model.fit_generator(train_generator,
#                     steps_per_epoch=4794//32,
#                     validation_data=validation_generator,
#                     validation_steps=1874//32,
#                     callbacks=[mcp_save],
#                     epochs=50,
#                     verbose=1)