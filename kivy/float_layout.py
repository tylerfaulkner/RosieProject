from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label

from kivy.config import Config
import numpy as np

import cv2

from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.models import load_model



Window.clearcolor = (115/255, 29/255, 38/255, 1)
Window.minimum_height = 720
Window.minimum_width = 1280
Window.toggle_fullscreen()

modelFile = "./models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

model=load_model('./models/ENetB0_E30_B64_ImageNet.h5')


labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}

class MyFloatLayout(FloatLayout):
    pass

class CamApp(App):

    def __init__(self, **kwargs):
        super().__init__()
        self.capture = cv2.VideoCapture(0)

    def build(self):
        #self.stresstest()
        self.img1=Image(pos=(-280, 145))
        layout = MyFloatLayout()
        logo = Image(pos=(-680, 450), color=(1,1,1,0.5), source='rosielogo.png')
        ermlogo = Image(pos=(-80, 5), color=(1,1,1,1), size_hint=(0.29,0.29), source='logo.png')
        #layout.add_widget(ermlogo)
        layout.add_widget(self.img1)
        layout.add_widget(logo)
        info = Label(text="This project was created by the Rosie team for Data Science Practicum.  This project is a culmination of over 1000 hours of model training. "
                          +"This was made possible only through the power of Rosie.\n\nThis project was completed in 2022. The members of the Team included Emma Straszewski, Nick Dang, Tyler Faulkner, Nigel Nelson, and Michael Salgado.",
                     font_size='38sp', pos=(1350, 324), size_hint=(0.27, 0.667), text_size=(480, 715), valign='center', halign='left')
        layout.add_widget(info)

        title = Label(text='Rosie Emotional Recognition Project', font_size='90sp', pos=(140, -380))
        layout.add_widget(title)


        Clock.schedule_interval(self.update, 1.0/33.0)

        layout.add_widget(ermlogo)
        return layout

    def stresstest(self):
        # Passing frame into face detection model
        frame = cv2.imread("images/15-images.jpg")

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()

        # Getting height and width of frame
        h, w = frame.shape[:2]

        # Input list to contain (numpy array of face, 4-element tuple of coordinates )
        input_list = []

        # For loop number of faces
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                # Getting sub image of face
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                sub_face_img = frame[y:y1, x:x1]

                # If face sub image is not empty, resize and reshape into numpy array
                # to pass into input list
                if sub_face_img.shape[0] != 0 and sub_face_img.shape[1] != 0:
                    resized = cv2.resize(sub_face_img, (224, 224))
                    reshaped = np.reshape(resized, (1, 224, 224, 3))
                    input_list.append((reshaped, (x, y, x1, y1)))

        # Check if input list is not empty
        if input_list:
            # Creating numpy array of each face sub image from input list
            faces_list = np.asarray([i[0] for i in input_list])
            for i in range(1, len(faces_list)):
                new_list = faces_list[:i]
            # Check if faces_list is not empty
                if np.any(new_list):
                    # Pass in tensor of faces into model for predictions
                    # result = model.predict(tf.convert_to_tensor(np.squeeze(faces_list, axis=1), dtype=tf.float16))
                    dataset = tf.data.Dataset.from_tensors(
                        tf.convert_to_tensor(np.squeeze(new_list, axis=1), dtype=tf.float16))
                    result = model.predict(dataset)

    def update(self, dt):
        success, frame = self.capture.read()
        frame = cv2.flip(frame, 1)

        if success:
            # Passing frame into face detection model
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         1.0, (300, 300), (104.0, 117.0, 123.0))
            net.setInput(blob)
            faces = net.forward()

            # Getting height and width of frame
            h, w = frame.shape[:2]

            # Input list to contain (numpy array of face, 4-element tuple of coordinates )
            input_list = []

            # For loop number of faces
            for i in range(faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence > 0.5:
                    # Getting sub image of face
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    sub_face_img = frame[y:y1, x:x1]

                    # If face sub image is not empty, resize and reshape into numpy array
                    # to pass into input list
                    if sub_face_img.shape[0] != 0 and sub_face_img.shape[1] != 0:
                        resized = cv2.resize(sub_face_img, (224, 224))
                        reshaped = np.reshape(resized, (1, 224, 224, 3))
                        input_list.append((reshaped, (x, y, x1, y1)))

            # Check if input list is not empty
            if input_list:
                # Creating numpy array of each face sub image from input list
                faces_list = np.asarray([i[0] for i in input_list])

                # Check if faces_list is not empty
                if np.any(faces_list):
                    # Pass in tensor of faces into model for predictions
                    # result = model.predict(tf.convert_to_tensor(np.squeeze(faces_list, axis=1), dtype=tf.float16))
                    dataset = tf.data.Dataset.from_tensors(
                        tf.convert_to_tensor(np.squeeze(faces_list, axis=1), dtype=tf.float16))
                    result = model.predict(dataset)

                    # result = model.predict(np.squeeze(faces_list, axis=1).astype(np.float16))

                    # For loop number of predictions
                    for i in range(len(result)):
                        # Get box coordinates of face from input list
                        x, y, x1, y1 = input_list[i][1]

                        # Get highest prediction
                        label = np.argmax(result[i])

                        # Drawing box around face
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

                        # Drawing text for current emotion
                        scaling_font_size = ((x1 - x) * 0.0047)
                        font_size = scaling_font_size if scaling_font_size >= 0.5 else 0.5
                        text_width = cv2.getTextSize(labels_dict[label], cv2.FONT_HERSHEY_SIMPLEX, font_size, 4)[0]
                        rectangle_mid = int((x1 - x) / 2)
                        cv2.putText(frame, labels_dict[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 4)
                        cv2.putText(frame, labels_dict[label], (((rectangle_mid + x) - (text_width[0] // 2)), y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)

            frame = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            frame = cv2.flip(frame, 0)

            buf = frame.tostring()

            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture1

if __name__ == '__main__':
    Config.set('graphics', 'width', '1280')
    Config.set('graphics', 'height', '720')
    Config.set('graphics', 'fullscreen', 'auto')
    Config.set('graphics', 'window_state', 'maximized')
    CamApp().run()
    cv2.destroyAllWindows()