# Used to perform a hyper parameter search for optimal values

# Uncomment below if unable to import "keras_tuner"
#%pip install keras-tuner --upgrade
import keras_tuner as kt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.keras.applications.efficientnet import EfficientNetB0
from tensorflow.python.client import device_lib

####################################################################
# Used to verify all expected GPUs are available
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print('GPUS:')
print(get_available_gpus())
print()
####################################################################

batch_size = 128
input_shape = (224,224)
target_size=(224,224)
train_dir = '/home/nelsonni/data/data/cs3310/Huupe/train_class'
test_dir = '/home/nelsonni/data/data/cs3310/Huupe/val_class'

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    rotation_range = 5,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

traindf=pd.read_csv('/data/cs3310/Huupe/affect_full_train.csv',dtype=str)
train_gen = tf.keras.preprocessing.image.ImageDataGenerator()
train_generator = train_gen.flow_from_dataframe(traindf, x_col='path', y_col='label', batch_size=batch_size,class_mode="categorical",target_size=target_size)

valdf = pd.read_csv('/data/cs3310/Huupe/affect_full_val.csv')
val_gen = tf.keras.preprocessing.image.ImageDataGenerator()
validation_generator = val_gen.flow_from_dataframe(valdf, x_col='path', y_col='label', batch_size=batch_size,class_mode="categorical",target_size=target_size)


def build_model(hp):  
    """
    Method used by the tuner to build models
    param: hp: the hyperparameters
    """
    # Create the keras model
    model = EfficientNetB0(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=(224, 224, 3),
    # Specifies the range of values for the tuner to try
    pooling=hp.Choice("pooling", ['None', 'avg', 'max']),
    classes=8,
    classifier_activation='softmax',
    )
    
    # Convert model to run on multiple GPUs
    parallel_model = multi_gpu_model(model, gpus=8)
    
    parallel_model.compile(
        tf.keras.optimizers.Adam(
            # Specifies the range of values for the tuner to try
            learning_rate= hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log"),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam"),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return parallel_model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30,
    hyperband_iterations=2,
    overwrite=True,
    directory="hyper_param_search",
    project_name='project_name')

tuner.search(train_generator,
             validation_data=validation_generator,
             epochs=30,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)],
             verbose=2)

# Save the best model from the hyperparameter search
best_model = tuner.get_best_models(1)[0]
best_model.save('best_ENetModel.h5')

# Print the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
print(best_hyperparameters)
print()

# Print the results of the hyperparameter search
print(tuner.results_summary())