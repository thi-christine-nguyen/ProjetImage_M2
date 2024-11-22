from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.camera import Camera
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.utils import platform
from kivy.clock import Clock
import numpy
from PIL import Image


# #-- maximize first, to get the screen size, minus any OS toolbars
Window.maximize()
maxSize = Window.system_size

# #-- set the actual window size, to be slightly smaller than full screen
# desiredSize = (maxSize[0]*0.9, maxSize[1]*0.9)
# Window.size = desiredSize

class TestCameraApp(App):
    def build(self):
        if platform == 'android':
            import android
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA])
        box=BoxLayout(orientation='vertical')
        self.mycam=Camera(play=False, index=0)
        #box.add_widget(self.mycam)
        
        self.w = Widget(pos=(0,0), size = maxSize)
        box.add_widget(self.w)
        tb=ToggleButton(text='Play', size_hint_y= None, height= '48dp')
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
            numpypicture=numpy.array(pil_image)
            print(numpypicture.shape)
            print(format(numpypicture))
            with self.w.canvas:
                Color(1, 1, 1)
                Rectangle(texture = texture, size=maxSize)

    def play(self, instance):
        if instance.state=='down':
            self.mycam.play=True
            instance.text='Stop'
        else:
            self.mycam.play=False
            instance.text='Play'
        
TestCameraApp().run()