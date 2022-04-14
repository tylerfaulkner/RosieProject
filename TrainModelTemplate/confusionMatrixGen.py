### Use the below srun command on rosie to run this script
### srun --partition=teaching --gpus=1 --cpus-per-gpu=2 singularity exec --nv -B /data:/data /data/containers/msoe-tensorflow-20.07-tf2-py3.sif /usr/local/bin/nvidia_entrypoint.sh python confusionMatrixGen.py


from tensorflow.keras.models import load_model
import seaborn as sns
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

batch_size = 1
input_shape = (224,224)
target_size=(224,224)

# test_dir = '/home/nelsonni/data/data/cs3310/Huupe/val_class'

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# validation_generator=test_datagen.flow_from_directory(test_dir,
#                                             class_mode="categorical",
#                                             target_size=input_shape,
#                                             batch_size=batch_size,
#                                             shuffle=False)

# num_test_imgs = 0
# for root, dirs, files in os.walk(test_dir):
#     num_test_imgs += len(files)

valdf = pd.read_csv('/data/cs3310/Huupe/affect_full_val.csv')

validation_generator = test_datagen.flow_from_dataframe(
    valdf,
    x_col='path',
    y_col='label',
    batch_size=batch_size,
    class_mode="categorical",
    target_size=target_size)


model=load_model('ENetB0_E45_B64_ImageNet_5149_valAcc.h5')

test_pred = model.predict(validation_generator, batch_size=batch_size)

test_pred = np.argmax(test_pred, axis=1)

# Calculate the confusion matrix using sklearn.metrics

cf_matrix = confusion_matrix(validation_generator.classes, test_pred)

labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

ax = sns.heatmap(cf_matrix,
                 annot=True,
                 cmap='Blues',
                 xticklabels=labels,
                 yticklabels=labels
                )

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

plt.savefig('ConfusionMatrix.png')

