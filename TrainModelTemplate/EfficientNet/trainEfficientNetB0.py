from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0
from tensorflow.python.client import device_lib
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint


### Double check that expected GPUs are available   #####

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print('GPUS:')
print(get_available_gpus())
print()

###########################################################

batch_size = 64  # Ensure batch size is divisible by number of GPUs available
epochs = 30
input_shape = (224,224)
target_size=(224,224)

###########################################################

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 30,
    zca_whitening=True,
    brightness_range=(0.5,1),
    channel_shift_range=0.4,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

traindf=pd.read_csv('/data/cs3310/Huupe/affect_full_train.csv',dtype=str)
valdf = pd.read_csv('/data/cs3310/Huupe/affect_full_val.csv')

# Remove the Contempt and Disgust rows

trainIndexOfContemptRows = traindf[traindf['label'] == 'Contempt'].index
traindf.drop(trainIndexOfContemptRows, inplace=True)
trainIndexOfDisgustRows = traindf[traindf['label'] == 'Disgust'].index
traindf.drop(trainIndexOfDisgustRows, inplace=True)

valIndexOfContemptRows = valdf[valdf['label'] == 'Contempt'].index
valdf.drop(valIndexOfContemptRows, inplace=True)
valIndexOfDisgustRows = valdf[valdf['label'] == 'Disgust'].index
valdf.drop(valIndexOfDisgustRows, inplace=True)

# Create the image generators
train_generator = train_datagen.flow_from_dataframe(traindf, x_col='path', y_col='label', batch_size=batch_size,class_mode="categorical",target_size=target_size)
validation_generator = test_datagen.flow_from_dataframe(valdf, x_col='path', y_col='label', batch_size=batch_size,class_mode="categorical",target_size=target_size)

# Code below ensures all available GPUs are used for training

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():

    # Create base model with ImageNet Weights
    base_model = EfficientNetB0(
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
    dropout = 0.3
    x = Dropout(dropout)(x)
    predictions = Dense(6, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        tf.keras.optimizers.Adam(
            learning_rate=0.00001,
            beta_1=0.9,
            beta_2=0.99999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam"),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

# Compute class weights according to sizes of the emotion categories
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_generator.classes), 
            train_generator.classes)

class_weights = dict(enumerate(class_weights))

# Create a checkpoint such that after each epoch if the validation accuracy is increased
# the model is saved
filepath = '6_class_with_aug_with_weigths.h5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_accuracy',
                             verbose=1, 
                             save_best_only=True,
                             mode='max')

history = model.fit(train_generator,
                    steps_per_epoch=traindf.shape[0] // batch_size + 1,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=valdf.shape[0] // batch_size + 1,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=2)


# save last epoch
model.save('6_class_with_aug_with_weigths_final_epoch.h5')

# save training history
np.save('6_class_with_aug_with_weigths.npy', history.history)

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
