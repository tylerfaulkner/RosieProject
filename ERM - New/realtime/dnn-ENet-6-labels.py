import cv2
import numpy as np
import time
from tensorflow.python.keras.utils import multi_gpu_utils
# from keras.models import load_model
from tensorflow.keras.models import load_model

# Loading the emotion recognition model
model=load_model('../models/ENetB0_6Class_ValAcc5780.h5')

# Setting video capture device
video=cv2.VideoCapture(0)

# Loading face detection model (OpenCV DNN)
modelFile = "../models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "../models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Setting labels dictionary
labels_dict={0:'Angry',1:'Fear',2:'Happy',3:'Neutral',4:'Sad',5:'Surprise'}

while True:
    # Reading video feed in by frame
    ret,frame=video.read()
    frame = cv2.flip(frame, 1)

    # Passing frame into face detection model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    h, w = frame.shape[:2]
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            # Getting sub image of face
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            sub_face_img = frame[y:y1, x:x1]

            try:
                # Reshaping sub image to pass into OpenCV DNN model
                resized = cv2.resize(sub_face_img, (224, 224))
                reshaped = np.reshape(resized, (1, 224, 224, 3))

                # Passing into model to predict emotion
                start = time.time()
                result = model.predict(reshaped / 255)
                end = time.time()
                print(f"inference time: {round(end - start, 2)} seconds")
                label = np.argmax(result, axis=1)[0]
                top_3_ind = np.argsort(-result[0])[:3]

                # Drawing box around face and text for current emotion
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


            except Exception as e:
                print(str(e))
    # Showing frame in window
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()