import cv2
import numpy as np
from keras.models import load_model
import os

model=load_model('../models/ENetB0_E30_B64_ImageNet.h5')

modelFile = "../models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "../models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}

# len(number_of_image), image_height, image_width, channel

frame=cv2.imread("../images/test_imge.png")
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
        resized = cv2.resize(sub_face_img, (224, 224))
        reshaped = np.reshape(resized, (1, 224, 224, 3))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # filename = "image-" + str(i) + ".jpg"
        # cv2.imwrite(filename, resized)
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x1, y1), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x1, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT, 0.8, (255, 255, 255), 2)
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)



cv2.imshow("Frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()