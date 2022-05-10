
# Uncomment below if missing modules 

# %pip install keras_applications
# %pip install --upgrade tensorflow
# %pip install --upgrade tensorflow-gpu
# %pip install keras-tuner --upgrade
# %pip install -U scikit-image


import keras_tuner as kt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.eager import context
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.io as sio
import os
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.applications.efficientnet import EfficientNetB3
from tensorflow.python.client import device_lib


### Double check that expected GPUs are available   #####

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print('GPUS:')
print(get_available_gpus())
print()

###########################################################

batch_size = 64  # Ensure batch size is divisible by number of GPUs available
input_shape = (224,224)
target_size=(224,224)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.1,
    rotation_range = 10,
    height_shift_range=4,
    width_shift_range=4,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

traindf=pd.read_csv('/data/cs3310/Huupe/affect_full_train.csv',dtype=str)
valdf = pd.read_csv('/data/cs3310/Huupe/affect_full_val.csv')

# Remove the Contempt and Disgust rows

trainIndexOfContemptRows = traindf[traindf['label'] == 'Contempt'].index
trainIndexOfDisgustRows = traindf[traindf['label'] == 'Disgust'].index

traindf.drop(trainIndexOfContemptRows, inplace=True)
traindf.drop(trainIndexOfDisgustRows, inplace=True)

valIndexOfContemptRows = valdf[valdf['label'] == 'Contempt'].index
valIndexOfDisgustRows = valdf[valdf['label'] == 'Disgust'].index

valdf.drop(valIndexOfContemptRows, inplace=True)
valdf.drop(valIndexOfDisgustRows, inplace=True)



train_generator = train_datagen.flow_from_dataframe(traindf, x_col='path', y_col='label', batch_size=batch_size,class_mode="categorical",target_size=target_size)
validation_generator = test_datagen.flow_from_dataframe(valdf, x_col='path', y_col='label', batch_size=batch_size,class_mode="categorical",target_size=target_size)


# Code below ensures all available GPUs are used for training

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():

    # Create base model with ImageNet Weights

    base_model = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling=None
        )

    # Add a layer with dropout to convert the 1024 way softmax to 6 classes
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    dropout = 0.2
    x = Dropout(dropout)(x)
    predictions = Dense(6, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        tf.keras.optimizers.Adam(
            learning_rate=0.00007,
            beta_1=0.6,
            beta_2=0.7,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam"),
        loss='categorical_crossentropy',
        metrics=['accuracy'])


epochs = 30

history = model.fit(train_generator,
                    steps_per_epoch=traindf.shape[0] // batch_size + 1,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=valdf.shape[0] // batch_size + 1,
                    verbose=2)
times = time_callback.times
print(f'TIMES: {times}')

model.save('ENetB3_E30_LR00007_B64.h5')


# Plot the training history

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
