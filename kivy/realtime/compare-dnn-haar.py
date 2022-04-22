import cv2
import numpy as np
import time
from keras.models import load_model

model = load_model('../models/ENetB0_E30_B64_ImageNet.h5')

video = cv2.VideoCapture(0)

modelFile = "../models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "../models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

faceDetect = cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')

haar_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
dnn_dict = {0: 'Angry', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Neutral', 6: 'Sad', 7: 'Surprise'}
while True:
    ret, frame = video.read()
    dnn_frame = cv2.flip(frame, 1)
    haar_frame = dnn_frame.copy()

    gray = cv2.cvtColor(haar_frame, cv2.COLOR_BGR2GRAY)
    haar_faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    blob = cv2.dnn.blobFromImage(cv2.resize(dnn_frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    dnn_faces = net.forward()
    h, w = dnn_frame.shape[:2]

    for i in range(dnn_faces.shape[2]):
        confidence = dnn_faces[0, 0, i, 2]
        if confidence > 0.5:
            box = dnn_faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            sub_face_img = dnn_frame[y:y1, x:x1]
            try:
                resized = cv2.resize(sub_face_img, (224, 224))
                reshaped = np.reshape(resized, (1, 224, 224, 3))
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]

                cv2.rectangle(dnn_frame, (x, y), (x1, y1), (0, 0, 255), 1)
                cv2.rectangle(dnn_frame, (x, y), (x1, y1), (50, 50, 255), 2)
                cv2.rectangle(dnn_frame, (x, y - 40), (x1, y), (50, 50, 255), -1)
                cv2.putText(dnn_frame, dnn_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            except Exception as e:
                print(str(e))

    for x, y, w, h in haar_faces:
        # start_time = time.perf_counter()
        sub_face_img = haar_frame[y:y + h, x:x + w]
        try:
            resized = cv2.resize(sub_face_img, (224, 224))
            reshaped = np.reshape(resized, (1, 224, 224, 3))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            print(result)
            cv2.rectangle(haar_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(haar_frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(haar_frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(haar_frame, haar_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        except Exception as e:
            print(str(e))
        # end_time = time.perf_counter()
        # print(f"Elapsed time per frame: {end_time} seconds")

    cv2.imshow("DNN Frame", dnn_frame)
    cv2.imshow("Haar Frame", haar_frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()