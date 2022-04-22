import cv2
import numpy as np
import time
import math
from keras.models import load_model
import threading
from keras.utils import multi_gpu_utils
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model



model=load_model('../models/ENetB0_E30_B64_ImageNet.h5')

video=cv2.VideoCapture(1)

modelFile = "../models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "../models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}

counter = 0
def predict_label(sub_face_image):
    resized = cv2.resize(sub_face_img, (224, 224))
    reshaped = np.reshape(resized, (1, 224, 224, 3))

    # if counter % 5 == 0:
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    top_3_ind = np.argsort(-result[0])[:3]
    cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x1, y1), (50, 50, 255), 2)
    # cv2.rectangle(frame, (x, y - 100), (x1, y), (50, 50, 255), -1)
    cv2.rectangle(frame, (x, y - 40), (x1, y), (50, 50, 255), -1)

    #
    # i = 1
    # for ind in reversed(top_3_ind):
    #     gap = 10
    #     prob = str(labels_dict[ind]) + ": " + str(round(result[0][ind], 3))
    #     cv2.putText(frame, prob, (x, y - (gap * i)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    #     i += 3
    cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


while True:
    counter += 1
    ret,frame=video.read()
    frame = cv2.flip(frame, 1)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    h, w = frame.shape[:2]
    threads = []
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            sub_face_img = frame[y:y1, x:x1]
            try:
                t = threading.Thread(target=predict_label, args=(sub_face_img,))
                threads.append(t)
                t.start()
            except Exception as e:
                print(str(e))
    frame = cv2.resize(frame, (1920, 1080), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()