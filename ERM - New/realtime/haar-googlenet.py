import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model

# Loading the emotion recognition model
model=load_model('../models/google_trained3')

# Setting video capture device
video=cv2.VideoCapture(0)

# Loading face detection model (Haar-Cascade)
faceDetect=cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')

# Setting labels dictionary
labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}

while True:
    # Reading video feed in by frame
    ret,frame=video.read()
    frame = cv2.flip(frame, 1)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Passing face into face detection model
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)

    for x,y,w,h in faces:
        # Getting sub image of face
        sub_face_img=frame[y:y+h, x:x+w]

        # Reshaping sub image to pass into emotion recognition model
        resized=cv2.resize(sub_face_img,(224,224))
        reshaped=np.reshape(resized, (1, 224, 224, 3))

        # Passing into model to predict emotion
        result=model.predict(reshaped)[2]
        label=np.argmax(result, axis=1)[0]

        # Drawing box around face and text for current emotion
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    # Showing frame in window
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()