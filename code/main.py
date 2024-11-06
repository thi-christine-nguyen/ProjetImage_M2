from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.camera import Camera
from kivy.core.window import Window

class TestCameraApp(App):
   def build(self):
      box=BoxLayout(orientation='vertical')
      self.mycam=Camera(play=False)
      box.add_widget(self.mycam)
      tb=ToggleButton(text='Play', size_hint_y= None, height= '48dp')
      tb.bind(on_press=self.play)
      box.add_widget(tb)
      return box

   def play(self, instance):
      if instance.state=='down':
         self.mycam.play=True
         instance.text='Stop'
      else:
         self.mycam.play=False
         instance.text='Play'
         
TestCameraApp().run()