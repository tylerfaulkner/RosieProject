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

import cv2

from kivy.core.window import Window
from kivy.graphics import Color, Rectangle

Window.clearcolor = (115/255, 29/255, 38/255, 1)
Window.minimum_height = 720
Window.minimum_width = 1280
Window.toggle_fullscreen()

class CamApp(App):

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

        self.capture = cv2.VideoCapture(0)
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
        # display image from cam in opencv window
        ret, frame = self.capture.read()
        #cv2.imshow("CV2 Image", frame)
        frame = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()

        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer. 
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.img1.texture = texture1

if __name__ == '__main__':
    Config.set('graphics', 'width', '1280')
    Config.set('graphics', 'height', '720')
    Config.set('graphics', 'fullscreen', 'auto')
    Config.set('graphics', 'window_state', 'maximized')
    CamApp().run()
    cv2.destroyAllWindows()