import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0

# %matplotlib inline
# %pip install -U scikit-image

import scipy.io as sio
import os

batch_size = 32
input_shape = (224,224)
train_dir = '/home/nelsonni/data/data/cs3310/Huupe/train_class'
test_dir = '/home/nelsonni/data/data/cs3310/Huupe/val_class'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    rotation_range = 5,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

validation_generator=test_datagen.flow_from_directory(test_dir,
                                            class_mode="categorical",
                                            target_size=input_shape,
                                            batch_size=batch_size)

train_generator=train_datagen.flow_from_directory(train_dir,
                                            class_mode="categorical",
                                            target_size=input_shape,
                                            batch_size=batch_size)

model = EfficientNetB0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling=None,
    classes=8,
    classifier_activation='softmax',
)

num_train_imgs = 0
for root, dirs, files in os.walk(train_dir):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_dir):
    num_test_imgs += len(files)

epochs = 2

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=num_train_imgs // 32,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs // 32)

model.save('ENetB0_30Epochs.h5')



