import cv2
import numpy as np
import tensorflow as tf
import time
import math
from tensorflow.python.keras.utils import multi_gpu_utils
from keras.models import load_model
from keras.preprocessing import image


# model=load_model('../models/ENetB0_E30_B64_ImageNet.h5')
# model.compile(optimizer="adam", loss="mean_squared_error")
model = load_model('../models/google_trained3')

video=cv2.VideoCapture(0)

modelFile = "../models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "../models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}
# labels_dict={0:'Angry',1:'Fear',2:'Happy',3:'Neutral',4:'Sad',5:'Surprise'}

counter = 0
while True:
    start = time.time()
    counter += 1
    ret,frame=video.read()
    frame = cv2.flip(frame, 1)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    h, w = frame.shape[:2]
    input_list = []
    if faces.shape[2] >= 1:
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                sub_face_img = frame[y:y1, x:x1]
                if np.any(sub_face_img):
                    resized = cv2.resize(sub_face_img, (224, 224))
                    reshaped = np.reshape(resized, (1, 224, 224, 3))
                    input_list.append((reshaped, (x, y, x1, y1)))

    if input_list:
        faces_list = np.asarray([i[0] for i in input_list])
        if np.any(faces_list):
            if len(faces_list) >= 2:
                result = model.predict_on_batch(tf.convert_to_tensor(np.squeeze(faces_list), dtype=tf.float16))[0]
            else:
                result = model.predict_on_batch(tf.convert_to_tensor(np.squeeze(faces_list, axis=1), dtype=tf.float16))[0]
            for i in range(len(result)):
                x, y, x1, y1 = input_list[i][1]
                label = np.argmax(result[i])
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

                font_size = ((x1-x) * 0.0047)
                text_width = cv2.getTextSize(labels_dict[label], cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
                rectangle_mid = int((x1 - x) / 2)
                cv2.putText(frame, labels_dict[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 4)
                cv2.putText(frame, labels_dict[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
    # frame = cv2.resize(frame, (1366, 1080), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    end = time.time()
    print(f"inference time: {round(end - start, 2)} seconds")
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()

