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

from keras.models import load_model


Window.clearcolor = (115/255, 29/255, 38/255, 1)
Window.minimum_height = 720
Window.minimum_width = 1280
Window.toggle_fullscreen()

modelFile = "./models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

model=load_model('./models/ENetB0_E30_B64_ImageNet.h5')


labels_dict={0:'Angry',1:'Contempt',2:'Disgust',3:'Fear',4:'Happy',5:'Neutral',6:'Sad',7:'Surprise'}

class CamApp(App):

    def __init__(self, **kwargs):
        super().__init__()
        self.capture = cv2.VideoCapture(0)

    def build(self):
        self.img1=Image(pos=(-280, 145))
        layout = FloatLayout()
        logo = Image(pos=(-680, 450), color=(1,1,1,0.5), source='rosielogo.png')
        layout.add_widget(self.img1)
        layout.add_widget(logo)
        info = Label(text="This project was created by the Rosie team for Data Science Practicum.  This project is a culmination of over 1000 hours of model training."
                          +"This was made possible only through the power of Rosie.\n\nThis project was completed in 2022. The members of the Team included Emma Straszewski, Nick Dang, Tyler Faulkner, Nigel Nelson, and Michael Salgado.",
                     font_size='38sp', pos=(1350, 324), size_hint=(0.27, 0.667), text_size=(480, 715), valign='center', halign='left')
        layout.add_widget(info)

        title = Label(text='Emotional Recognition Project', font_size='120sp', pos=(40, 20), size_hint=(0.951, 0.26))
        layout.add_widget(title)

        Clock.schedule_interval(self.update, 1.0/33.0)

        with title.canvas.before:
            Color(4/255, 6/255, 4/255, 1)  # green; colors range from 0-1 instead of 0-255
            title.rect = Rectangle(size=title.size,
                                  pos=title.pos)
        with info.canvas.before:
            Color(4/255,6/255,4/255,1)
            info.rect = Rectangle(size=info.size,pos=info.pos)

        def update_rect(instance, value):
            instance.rect.pos = instance.pos
            instance.rect.size = instance.size

        # listen to size and position changes
        title.bind(pos=update_rect, size=update_rect)
        info.bind(pos=update_rect, size=update_rect)
        return layout

    def update(self, dt):
        success, frame = self.capture.read()
        #frame = cv2.flip(frame, 1)
        if success:
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
                    #This line causes it to crash sometimes
                    #Sometimes crahses due to face not having and width only height in image
                    # EG (85,0,3)
                    if sub_face_img.shape[0] != 0 and sub_face_img.shape[1] != 0:
                        resized = cv2.resize(sub_face_img, (224, 224))
                        reshaped = np.reshape(resized, (1, 224, 224, 3))

                        result = model.predict(reshaped)
                        label = np.argmax(result, axis=1)[0]
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 1)
                        cv2.rectangle(frame, (x, y), (x1, y1), (50, 50, 255), 2)
                        cv2.rectangle(frame, (x, y - 40), (x1, y), (50, 50, 255), -1)

                        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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