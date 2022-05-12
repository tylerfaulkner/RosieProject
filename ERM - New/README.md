ERM folder
--------------------

This folder contains much of the soruce code and model files used to crete our minimum viable product.

The MVP only doesn't have a UI and only uses the models and draws the boxes and predicted emotion around the segmented face.

Description of files
--------------------

Non-Python files:

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
README.md                         |  Text file (markdown format) description of the folder


Data folder:

This folder contained the images from the FER-2013 data set and was used to train models using scripts from this repository (./training/main.py). 
These images were removed from the repository as we switched to the AffectNet data set, but the folder is retained for future use.

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------


Images folder:

This folder contains images that can be used in the scripts contained in the run_image folder. The scripts can read in an image from this folder
and predict the subjects' emotions. Each image was obtained through Google images.

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
8-Figure3-1.png  |   A single image containing 12 faces from the AffectNet dataset. Each face contains their predicted emotion from a model pass online, as well as the true label.
emotion_test_1.jpeg                        |  A single image containing 7 faces on a solid green background. Each face contains their true emotion. 

Models folder:

This folder contains model files that were used throughout our various implementations. These include emotion detection models that were built using Rosie as well as face detection files
that were obtained through public libraries and repositories.

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
/google_trained3  |    This folder contains the emotion recognition model using GoogleNet. The model was saved using the SavedModel format from Keras where a folder contains the models assets, variables, and a .pb file which contains the layer information. 
/saved_model                        |  
baseline_affectnet_sample_30epochs.h5                        |  This baseline model uses the same layers and setup as an online tutorial on building a model for emotion recognition. This iteration was trained on a sample AffectNet dataset using Rosie for 30 epochs.
baseline_fer_model_files_30epochs.h5                        |  This baseline model was trained on the FER-2013 dataset using Rosie for 30 epochs.
baseline_fer_model_file_100epochs.h5                        |  This baseline model was trained on the FER-2013 dataset using Rosie for 100 epochs.
deploy.prototxt.txt                        |  A required file for the OpenCV DNN face detection model 
ENetB0_6Class_ValAcc5780.h5                        |  This EfficientNet model was trained on the AffectNet dataset using Rosie for 30 epochs. Two categories (contempt and disgust) were pruned from this models training in order to increase our accuracy.
ENetB0_E30_b64_ImageNet.h5                        |   This EfficientNet model was trained on the AffectNet dataset using Rosie for 30 epochs.
haarcascade_frontalface_default.xml                        |  The Haar-Cascade face detection model.
res10_300x300_ssd_iter_140000.caffemodel                        |  A required file for the OpenCV DNN face detection model.

Realtime folder:

This folder contains our python scripts that are used for realtime emotion recognition. Each file contains a general implementation of obtaining a video feed, passing the frame into a face detection model, and then passing in each face sub image into an emotion
recognition model to predict a subject's emotion and draw a box around their face as well as their emotion label. 

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
/multi_thread_MVP  |    
compare-dnn-haar.py                        |  File to compare the performance of the haar-cascade and OpenCV DNN face detection models. Both implementations take in the same video frame and pass them into the same model, which gives an equal comparison for face detection.
dnn-ENet.py                        |  A script that uses the OpenCV DNN face detection model and the 8 class EfficientNet model to detect facial emotion in real time
dnn-ENet-6-labels.py                        |  A script that uses the OpenCV DNN face detection model and the 6 class EfficientNet model to detect facial emotion in real time
dnn-ENet-optimization.py                        |  A script that uses the OpenCV DNN face detection model and the 8 class EfficientNet model to detect facial emotion in real time. This script was optimized for lower latency by passing in each face sub image into a tensor, which is then passed into the emotion recognition model all at once rather than passing them in one at a time.
dnn_baseline.py                        |  A script that uses the OpenCV DNN face detection model and the baseline model to detect facial emotion in real time
haar-ENet.py                        |   A script that uses the haar-cascade face detection model and the 8 class EfficientNet model to detect facial emotion in real time
haar-googlenet.py                        |   A script that uses the haar-cascade face detection model and the GoogLeNet model to detect facial emotion in real time

Run_image folder:

This folder contains our python scripts that are used for a single pass of emotion recognition. Each file contains a general implementation of taking in an image, passing the frame into a face detection model, and then passing in each face sub image into an emotion
recognition model to predict a subject's emotion and draw a box around their face as well as their emotion label. 

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
dnn-ENet-singleimage.py |    The script for passing in a single image into the OpenCV DNN face detection model and then passing in each face sub image into the EffecientNet model
haar-ENet-singleimage.py |    The script for passing in a single image into the haar-cascade face detection model and then passing in each face sub image into the EffecientNet model

Training folder:

This folder contains python scripts that were used for the baseline model training.

filename                          |  description
----------------------------------|------------------------------------------------------------------------------------
main.py |    The baseline model training script




