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
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Découpage d'image avec dlib
def dlib_cut(image):
    if image is not None:
        im = image.copy()
        
        # Convertir en niveaux de gris (requis pour Dlib)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Détection des visages
        faces = detector(gray)
        
        if len(faces) > 0:
            print("faces")
            # Dessiner un rectangle autour des visages et afficher les landmarks
            for face in faces:
                print("face")
                x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                # Dessiner un rectangle autour du visage
                # cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
                
                # Extraire les landmarks pour chaque visage détecté
                landmarks = predictor(gray, face)
                
                # # Dessiner les points de landmarks
                # for i in range(0, 68):  # 68 points de landmarks
                #     x_landmark = landmarks.part(i).x
                #     y_landmark = landmarks.part(i).y
                #     cv2.circle(image, (x_landmark, y_landmark), 1, (255, 0, 0), -1)
            
            # Prendre le premier visage détecté
            first_face = faces[0]
            x, y, x2, y2 = first_face.left(), first_face.top(), first_face.right(), first_face.bottom()
            w, h = x2 - x, y2 - y
            print("etape 1")
            
            # Calculer les dimensions du cadre centré autour du visage
            center_x = x + w / 2
            center_y = y + h / 2
            height, width, channels = im.shape
            print("etape 2")
            
            # Ajustement pour inclure du contexte autour du visage
            margin = 0.4  # Marge contextuelle (40% des dimensions du visage)
            box_w = w * (1 + margin)
            box_h = h * (1 + margin)
            box_dim = min(box_w, box_h, width, height)  # Limiter la taille à l'image
            print("etape 3")
            
            # Calculer les coordonnées du cadre centré
            x_start = max(0, int(center_x - box_dim / 2))
            y_start = max(0, int(center_y - box_dim / 2))
            x_end = min(width, int(center_x + box_dim / 2))
            y_end = min(height, int(center_y + box_dim / 2))
            print("etape 4")
            
            # Découper et redimensionner l'image
            crpim = im[y_start:y_end, x_start:x_end]
            crpim = cv2.resize(crpim, (224, 224), interpolation=cv2.INTER_AREA)
            print("etape 5")
            
            # Retourner l'image découpée, l'image annotée et les dimensions du visage
            print(f"Found {len(faces)} faces!")
            return crpim, image, (x, y, w, h)
    
    return None, image, (0, 0, 0, 0)


# get Screen size
Window.maximize()
maxSize = Window.system_size

class TestCameraApp(App):
    def build(self):
        box=BoxLayout(orientation='vertical')
        self.mycam=Camera(play=False, index=0)
        #box.add_widget(self.mycam)
        self.w = Widget(pos=(0,0))
        box.add_widget(self.w)

        self.tb=ToggleButton(text=platform, size_hint_y= None, height= '48dp')
        self.tb.bind(on_press=self.play)
        box.add_widget(self.tb)
        self.task = Clock.schedule_interval(self.updateCanvas, 0.1)  
        return box

    def updateCanvas(self, dt):
        if self.mycam.texture != None:
            #  Extrait l'image de la camera
            texture = self.mycam.texture
            size=texture.size
            print(size)
            pixels = texture.pixels
            pil_image=Image.frombytes(mode='RGBA', size=size,data=pixels)

            # Rotate image if on android
            if platform == 'android':
                pil_image=pil_image.rotate(90, expand=True)
                texture = Texture.create(size=(texture.size[1], texture.size[0]), colorfmt='rgba')
            else:
                pil_image=pil_image.rotate(180)
                texture = Texture.create(size=(texture.size[0], texture.size[1]), colorfmt='rgba')
                
            open_cv_image = numpy.array(pil_image)
            self.tb.text= str(dlib_cut(open_cv_image)[2])
            #Calc image position
            textureRatio = max(min(maxSize[0]//texture.size[0], maxSize[1]//texture.size[1]),1)
            textureSize = (texture.size[0] * textureRatio, texture.size[1] * textureRatio)
            screenPosition = ((maxSize[0] - textureSize[0])//2, (maxSize[1] - textureSize[1])//2)
            # show images
            texture.blit_buffer(pil_image.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
            self.w.canvas.clear()
            with self.w.canvas:
                Color(1, 1, 1)
                Rectangle(texture = texture, size=textureSize, pos = screenPosition)


    def play(self, instance):
        if instance.state=='down':
            self.mycam.play=True
            instance.text='Stop'
        else:
            self.mycam.play=False
            instance.text='Play'
         
TestCameraApp().run()