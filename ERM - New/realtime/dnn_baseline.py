import cv2
import numpy as np
import time
from keras.models import load_model
from mtcnn.mtcnn import MTCNN

model=load_model('../models/baseline_affectnet_sample_30epochs.h5')

video=cv2.VideoCapture(0)

modelFile = "../models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "../models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    height, width = frame.shape[:2]
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            resized = cv2.resize(gray, (48, 48))
            reshaped = np.reshape(resized, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            print(labels_dict[label])
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()