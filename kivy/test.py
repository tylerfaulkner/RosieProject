from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label

import cv2

class CamApp(App):

    def build(self):
        self.img1=Image(size_hint=(0.6, 1.0))
        layout = BoxLayout(orientation='vertical')
        topRow = BoxLayout(orientation='horizontal')
        bottomRow = BoxLayout(orientation='horizontal', size_hint=(1.0,0.2))

        layout.add_widget(topRow)
        layout.add_widget(bottomRow)

        topRow.add_widget(Label(text='Left', font_size='40sp', size_hint=(0.2, 1.0)))
        topRow.add_widget(self.img1)
        topRow.add_widget(Label(text='Right', font_size='40sp', size_hint=(0.2, 1.0)))

        bottomRow.add_widget(Label(text='Emotional Recognition Project', font_size='40sp'))
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        #cv2.namedWindow("CV2 Image")
        Clock.schedule_interval(self.update, 1.0/33.0)
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
    CamApp().run()
    cv2.destroyAllWindows()