from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.camera import Camera
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.graphics import opengl
from kivy.clock import Clock
from kivy.utils import platform

import numpy
from PIL import Image

# #-- maximize first, to get the screen size, minus any OS toolbars
# Window.maximize()
# maxSize = Window.system_size

# #-- set the actual window size, to be slightly smaller than full screen
# desiredSize = (maxSize[0]*0.9, maxSize[1]*0.9)
# Window.size = desiredSize

class TestCameraApp(App):
    def build(self):
        box=BoxLayout(orientation='vertical')
        self.mycam=Camera(play=False, index=0)
        #box.add_widget(self.mycam)
        self.w = Widget(pos=(0,0))
        box.add_widget(self.w)

        tb=ToggleButton(text=platform, size_hint_y= None, height= '48dp')
        tb.bind(on_press=self.play)
        box.add_widget(tb)
        self.task = Clock.schedule_interval(self.updateCanvas, 0.1)  
        return box

    def updateCanvas(self, dt):
        if self.mycam.texture != None:
            texture = self.mycam.texture
            size=texture.size
            pixels = texture.pixels
            pil_image=Image.frombytes(mode='RGBA', size=size,data=pixels)

            if platform == 'android':
                pil_image=pil_image.rotate(90, expand=True)
                texture = Texture.create(size=(texture.size[1], texture.size[0]), colorfmt='rgba')
            else:
                pil_image=pil_image.rotate(180)
                texture = Texture.create(size=(texture.size[0], texture.size[1]), colorfmt='rgba')

            texture.blit_buffer(pil_image.tobytes(), colorfmt='rgba', bufferfmt='ubyte')


            # numpypicture=numpy.array(pil_image)
            # print(numpypicture.shape)
            # print(format(numpypicture))
            self.w.canvas.clear()
            with self.w.canvas:
                Color(1, 1, 1)
                Rectangle(texture = texture, size=(800,800))


    def play(self, instance):
        if instance.state=='down':
            self.mycam.play=True
            instance.text='Stop'
        else:
            self.mycam.play=False
            instance.text='Play'
         
TestCameraApp().run()