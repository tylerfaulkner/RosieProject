import cv2
import numpy as np
import time
import math
from tensorflow.python.keras.utils import multi_gpu_utils
from keras.models import load_model
from keras.preprocessing import image


model=load_model('../models/ENetB0_Acc6313.h5')
# model.compile(optimizer="adam", loss="mean_squared_error")
# model = load_model('../models/google_trained3')

video=cv2.VideoCapture(0)

modelFile = "../models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "../models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}
labels_dict={0:'Angry',1:'Fear',2:'Happy',3:'Neutral',4:'Sad',5:'Surprise'}

counter = 0
while True:
    counter += 1
    ret,frame=video.read()
    frame = cv2.flip(frame, 1)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    h, w = frame.shape[:2]
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            sub_face_img = frame[y:y1, x:x1]
            try:
                resized = cv2.resize(sub_face_img, (224, 224))
                reshaped = np.reshape(resized, (1, 224, 224, 3))

                # if counter % 2 == 0:
                start = time.time()
                result = model.predict(reshaped)
                # result = model.predict(reshaped, batch_size=len(reshaped))
                end = time.time()
                print(f"inference time: {round(end - start, 2)} seconds")
                label = np.argmax(result, axis=1)[0]
                top_3_ind = np.argsort(-result[0])[:3]

                cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                # cv2.rectangle(frame, (x, y - 100), (x1, y), (50, 50, 255), -1)
                # cv2.rectangle(frame, (x, y - 40), (x1, y), (50, 50, 255), -1)

                #
                # i = 1
                # for ind in reversed(top_3_ind):
                #     gap = 10
                #     prob = str(labels_dict[ind]) + ": " + str(round(result[0][ind], 3))
                #     cv2.putText(frame, prob, (x, y - (gap * i)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                #     i += 3
                cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


            except Exception as e:
                print(str(e))
    # frame = cv2.resize(frame, (1366, 768), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()