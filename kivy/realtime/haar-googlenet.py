import cv2
import numpy as np
import time
from keras.models import load_model

model=load_model('../models/google_trained3')

video=cv2.VideoCapture(0)

faceDetect=cv2.CascadeClassifier('../models/haarcascade_frontalface_default.xml')

labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}

while True:
    ret,frame=video.read()
    frame = cv2.flip(frame, 1)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces= faceDetect.detectMultiScale(gray, 1.3, 3)

    for x,y,w,h in faces:
        # start_time = time.perf_counter()
        sub_face_img=frame[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(224,224))
        reshaped=np.reshape(resized, (1, 224, 224, 3))
        result=model.predict(reshaped)[2]
        label=np.argmax(result, axis=1)[0]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        # end_time = time.perf_counter()
        # print(f"Elapsed time per frame: {end_time} seconds")
        
    cv2.imshow("Frame",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()