#%pip install keras_applications
import keras_tuner as kt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import multi_gpu_model
#from tensorflow.python.keras.applications.efficientnet import EfficientNetB0

# %matplotlib inline
# %pip install -U scikit-image

batch_size = 64
input_shape = (224,224)
target_size=(224,224)
train_dir = '/home/nelsonni/data/data/cs3310/Huupe/train_class'
test_dir = '/home/nelsonni/data/data/cs3310/Huupe/val_class'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.1,
    rotation_range = 10,
    height_shift_range=4,
    width_shift_range=4,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# validation_generator=test_datagen.flow_from_directory(test_dir,
#                                             class_mode="categorical",
#                                             target_size=input_shape,
#                                             batch_size=batch_size)

# train_generator=train_datagen.flow_from_directory(train_dir,
#                                             class_mode="categorical",
#                                             target_size=input_shape,
#                                             batch_size=batch_size)

traindf=pd.read_csv('/data/cs3310/Huupe/affect_full_train.csv',dtype=str)
# train_gen = tf.keras.preprocessing.image.ImageDataGenerator()
train_generator = train_datagen.flow_from_dataframe(traindf, x_col='path', y_col='label', batch_size=batch_size,class_mode="categorical",target_size=target_size)

valdf = pd.read_csv('/data/cs3310/Huupe/affect_full_val.csv')
#val_gen = tf.keras.preprocessing.image.ImageDataGenerator()
validation_generator = test_datagen.flow_from_dataframe(valdf, x_col='path', y_col='label', batch_size=batch_size,class_mode="categorical",target_size=target_size)

# model = MobileNetV2(
#     input_shape=(224, 224, 3),
#     alpha=1.0,
#     include_top=True,
#     weights=None,
#     input_tensor=None,
#     pooling=None,
#     classes=8
# )

# parallel_model = multi_gpu_model(model, gpus=4)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
        )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    dropout = 0.5
    x = Dropout(dropout)(x)
    predictions = Dense(8, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
    tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam"),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# def build_model(hp): 
    
#     model = MobileNetV2(
#     input_shape=(224, 224, 3),
#     alpha= hp.Float("alpha", min_value=.5, max_value=1.5, step=.5),
#     include_top=True,
#     weights=None,
#     input_tensor=None,
#     pooling=hp.Choice("pooling", ['None', 'avg', 'max']),
#     classes=8
#     )
    
#     parallel_model = multi_gpu_model(model, gpus=4)
    
#     parallel_model.compile(
#         tf.keras.optimizers.Adam(
#             learning_rate= hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log"),
#             beta_1=0.9,
#             beta_2=0.999,
#             epsilon=1e-07,
#             amsgrad=False,
#             name="Adam"),
#         loss='categorical_crossentropy',
#         metrics=['accuracy'])

    
#     return parallel_model

# tuner = kt.Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=30,
#     hyperband_iterations=2,
#     overwrite=True,
#     directory="hyper_param_search_mobile",
#     project_name='project_name')

# tuner.search(train_generator,
#              validation_data=validation_generator,
#              epochs=30,
#              callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
#             verbose=2)

# best_model = tuner.get_best_models(1)[0]
# best_model.save('best_MobileNetModel.h5')

# best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
# print(best_hyperparameters)
# print()
# print(tuner.results_summary())

num_train_imgs = 0
for root, dirs, files in os.walk(train_dir):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_dir):
    num_test_imgs += len(files)

epochs = 45

# model.compile(
#     tf.keras.optimizers.Adam(
#         learning_rate=0.00001,
#         beta_1=0.9,
#         beta_2=0.999,
#         epsilon=1e-07,
#         amsgrad=False,
#         name="Adam"),
#     loss='categorical_crossentropy',
#     metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=num_train_imgs // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs // batch_size,
                    verbose=2)

model.save('MobileNet_E45_B64_LR0001l.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
