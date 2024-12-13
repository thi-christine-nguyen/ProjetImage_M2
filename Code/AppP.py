from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock

import os
import cv2
import numpy as np
import dlib
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Flatten, Dropout, Activation, Permute
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
K.set_image_data_format( 'channels_last' )
from scipy.spatial.distance import cosine as dcos
from scipy.io import loadmat
import pickle

# Variables globales
current_decoupe_method = "dlib_cut"
detection_threshold = 0.3

# Initialisation des modèles Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def dlib_cut(image):
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        results = []

        for face in faces:
            x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            w, h = x2 - x, y2 - y
            center_x = x + w / 2
            center_y = y + h / 2
            height, width, channels = image.shape

            margin = 0.4  # Marge contextuelle
            box_w = w * (1 + margin)
            box_h = h * (1 + margin)
            box_dim = min(box_w, box_h, width, height)

            x_start = max(0, int(center_x - box_dim / 2))
            y_start = max(0, int(center_y - box_dim / 2))
            x_end = min(width, int(center_x + box_dim / 2))
            y_end = min(height, int(center_y + box_dim / 2))

            cropped_face = image[y_start:y_end, x_start:x_end]
            cropped_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)
            results.append((cropped_face, (x, y, w, h)))

        return results, image
    return [], image

# Modification de haar
def haar(image):
    if image is not None:
        if not hasattr(haar, 'faceCascade'):
            haar.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = haar.faceCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        results = []

        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            height, width, _ = image.shape
            b_dim = min(max(w, h) * 1.2, width, height)
            box = [center_x - b_dim // 2, center_y - b_dim // 2,
                   center_x + b_dim // 2, center_y + b_dim // 2]
            box = [int(coord) for coord in box]

            if box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height:
                cropped_face = image[box[1]:box[3], box[0]:box[2]]
                cropped_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)
                results.append((cropped_face, (x, y, w, h)))

        return results, image
    return [], image

# Création du CNN
def convblock(cdim, nb, bits=3):
    L = []
    for k in range(1,bits+1):
        convname = 'conv'+str(nb)+'_'+str(k)
        L.append( Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname) )
    L.append( MaxPooling2D((2, 2), strides=(2, 2)) )
    return L

def vgg_face_blank():
    withDO = True
    if True:
        mdl = Sequential()
        mdl.add( Permute((1,2,3), input_shape=(224,224,3)) )
        for l in convblock(64, 1, bits=2):
            mdl.add(l)
        for l in convblock(128, 2, bits=2):
            mdl.add(l)        
        for l in convblock(256, 3, bits=3):
            mdl.add(l)            
        for l in convblock(512, 4, bits=3):
            mdl.add(l)            
        for l in convblock(512, 5, bits=3):
            mdl.add(l)        
        mdl.add( Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6') )
        if withDO:
            mdl.add( Dropout(0.5) )
        mdl.add( Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7') )
        if withDO:
            mdl.add( Dropout(0.5) )
        mdl.add( Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8') )
        mdl.add( Flatten() )
        mdl.add( Activation('softmax') )
        
        return mdl
    
    else:
        raise ValueError('not implemented')

# Utilisation de Vgg-Face
def copy_mat_to_keras(kmodel):
    kerasnames = [lr.name for lr in kmodel.layers]
    prmt = (0,1,2,3)

    for i in range(l.shape[1]):
        matname = l[0,i][0,0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            l_weights = l[0,i][0,0].weights[0,0]
            l_bias = l[0,i][0,0].weights[0,1]
            f_l_weights = l_weights.transpose(prmt)
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])

def save_database(database, file_path="database.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(database, f)
    print(f"Base de données sauvegardée dans {file_path}")

def load_database(file_path="database.pkl"):
    try:
        with open(file_path, "rb") as f:
            database = pickle.load(f)
        print(f"Base de données chargée depuis {file_path}")
        return database
    except FileNotFoundError:
        print(f"Aucune base de données trouvée à {file_path}.")
        return {}

def generate_database(folder_img="FaceDataBase", save_cropped_images=True, save_folder="CroppedImagesDlib"):
    # Démarrer le chronomètre
    #start_time = time.time()
    database = {}

    # Créer le dossier pour enregistrer les images découpées si nécessaire
    if save_cropped_images and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Parcours des sous-dossiers (chaque sous-dossier représente une personne)
    for person_folder in os.listdir(folder_img):
        person_folder_path = os.path.join(folder_img, person_folder)

        if os.path.isdir(person_folder_path):
            for img_file in os.listdir(person_folder_path):
                img_path = os.path.join(person_folder_path, img_file)

                try:
                    if os.path.isfile(img_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = cv2.imread(img_path)

                        # Détection de plusieurs visages dans l'image
                        faces, _ = dlib_cut(img)

                        for i, (crpim, (x, y, w, h)) in enumerate(faces):
                            if crpim is not None:
                                # Enregistrer l'image recadrée si nécessaire
                                if save_cropped_images:
                                    cropped_img_path = os.path.join(
                                        save_folder, f"{person_folder}_{img_file.split('.')[0]}_face{i}.jpg"
                                    )
                                    cv2.imwrite(cropped_img_path, crpim)

                                # Extraction des caractéristiques de l'image avec le modèle
                                vector_image = crpim[None, ...]
                                # Envoi de l'image au CNN
                                person_vector = featuremodel.predict(vector_image)[0, :]

                                # Ajout du vecteur à la base de données
                                if person_folder not in database:
                                    database[person_folder] = []
                                database[person_folder].append(person_vector)

                except Exception as e:
                    print(f"Erreur avec l'image {img_path}: {e}")

    # Calculer le temps écoulé
    #elapsed_time = time.time() - start_time
    #print(f"Temps écoulé pour générer la base de données : {elapsed_time:.2f} secondes.")

    return database



# Fonction pour trouver la personne la plus proche en utilisant la base de données
def find_closest_angle(img, database, min_detection=0.3):
    imarr1 = np.asarray(img)
    imarr1 = imarr1[None, ...]
    
    # Prédiction du vecteur de caractéristiques de l'image
    fvec1 = featuremodel.predict(imarr1)[0, :]
    
    # Recherche de la personne la plus proche dans la base de données
    dmin = float('inf')
    umin = ""
    
    for person, vectors in database.items():
        for fvec2 in vectors:
            dcos_1_2 = dcos(fvec1, fvec2)
            if dcos_1_2 < dmin:
                dmin = dcos_1_2
                umin = person
    
    if dmin > detection_threshold:
        umin = "Inconnu"
    
    return umin, dmin

def find_closest_distance(img, database, min_detection=0.7):
    imarr1 = np.asarray(img)
    imarr1 = imarr1[None, ...]  # Ajouter une dimension pour le batch
    
    # Prédiction du vecteur de caractéristiques de l'image
    fvec1 = featuremodel.predict(imarr1)[0, :]
    
    # Normaliser le vecteur de caractéristiques (si vous utilisez la distance euclidienne)
    fvec1 = fvec1 / np.linalg.norm(fvec1)
    
    # Recherche de la personne la plus proche dans la base de données
    dmin = float('inf')
    umin = ""
    
    for person, vectors in database.items():
        for fvec2 in vectors:
            # Normaliser fvec2
            fvec2 = fvec2 / np.linalg.norm(fvec2)
            
            # Calcul de la distance euclidienne entre fvec1 et fvec2
            dist = np.linalg.norm(fvec1 - fvec2)
            
            if dist < dmin:
                dmin = dist
                umin = person
    
    # Si la distance minimale est supérieure à un seuil, retourner une personne inconnue
    if dmin > min_detection:
        print("haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        umin = "Inconnu"
    
    return umin, dmin


def recognize_image(imgcrop, database):
    name, dmin = find_closest_angle(imgcrop, database)
    print(name)
    return name, True


# Classe principale pour Kivy
class MainApp(App):
    def build(self):
        # Layout principal
        layout = BoxLayout(orientation="vertical")

        # Layouts secondaires
        imageLayout = BoxLayout(orientation="horizontal")
        buttonLayout = BoxLayout(orientation="vertical",size_hint=(0.2,0.9))
        decoupeLayout = GridLayout(cols=2)
        detectLayout = GridLayout(cols=2)
        tresholdLayout = BoxLayout(orientation="vertical", size_hint=(1,0.1))

        # Affichage vidéo
        self.video = Image(size_hint=(0.6,0.9))

        decoupe_label = Label(text="Detection\nde visage")
        dlib_label = Label(text="Dlib \nCut")
        haar_label = Label(text="Haar \nCascade")
        dlib_check = CheckBox(active=True, group="decoupe")
        dlib_check.bind(active=lambda _,__: self.set_decoupe_method("dlib_cut"))
        haar_check = CheckBox(active=False, group="decoupe")
        haar_check.bind(active=lambda _,__: self.set_decoupe_method("haar"))

        detect_label = Label(text="Reconnaissance\nde visage")
        vggFace_label = Label(text="VggFaces")
        # lbph_label = Label(text="Haar Cascade")
        vggFace_check = CheckBox(active=True, group="detect")
        # lbph_check = CheckBox(active=False, group="detect")

        # Boutons pour choisir la méthode
        # btn_dlib = Button(text="Dlib Cut", on_press=lambda x: self.set_method("dlib_cut"))
        # btn_haar = Button(text="Haar Cascade", on_press=lambda x: self.set_method("haar"))

        self.desc_label = Label(text="", size_hint=(0.2,0.9))
        # Curseur pour ajuster le seuil
        self.threshold_label = Label(text=f"Seuil: {detection_threshold:.2f}")
        self.threshold_slider = Slider(min=0.1, max=1.0, value=detection_threshold, step=0.01)
        self.threshold_slider.bind(value=self.update_threshold)

        # Ajouter les widgets dans les layouts
        tresholdLayout.add_widget(self.threshold_label)
        tresholdLayout.add_widget(self.threshold_slider)

    
        buttonLayout.add_widget(decoupe_label)
        decoupeLayout.add_widget(dlib_label)
        decoupeLayout.add_widget(dlib_check)
        decoupeLayout.add_widget(haar_label)
        decoupeLayout.add_widget(haar_check)
        buttonLayout.add_widget(decoupeLayout)
        buttonLayout.add_widget(detect_label)
        detectLayout.add_widget(vggFace_label)
        detectLayout.add_widget(vggFace_check)
        # detectLayout.add_widget(lbph_label)
        # detectLayout.add_widget(lbph_check)
        buttonLayout.add_widget(detectLayout)
        # buttonLayout.add_widget(btn_dlib)
        # buttonLayout.add_widget(btn_haar)

        imageLayout.add_widget(buttonLayout)
        imageLayout.add_widget(self.video)
        imageLayout.add_widget(self.desc_label)

        # Ajouter les sous-layouts dans le layout principal
        layout.add_widget(imageLayout)
        layout.add_widget(tresholdLayout)

        # Lancer le flux vidéo
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        self.frame_counter =0
        self.detected_faces = []

        return layout

    def set_decoupe_method(self, method):
        global current_decoupe_method
        current_decoupe_method = method

    def update_threshold(self, instance, value):
        global detection_threshold
        detection_threshold = value
        self.threshold_label.text = f"Seuil: {value:.2f}"

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Miroir pour une meilleure expérience utilisateur
            self.frame_counter += 1

            if self.frame_counter >= 5:
                self.frame_counter = 0  # Réinitialiser le compteur

                # Réaliser le traitement des visages toutes les 10 frames
                if current_decoupe_method == "dlib_cut":
                    faces, _ = dlib_cut(frame)
                elif current_decoupe_method == "haar":
                    faces, _ = haar(frame)
                else:
                    faces = []

                self.detected_faces = faces  # Mettre à jour les visages détectés

            # Annoter les résultats à chaque frame avec les données précédemment détectées
            for cropped_face, (x, y, w, h) in self.detected_faces:
                name, _ = recognize_image(cropped_face, db)
                try:
                    self.desc_label.text = name + "\n" + data_info[name]
                except:
                    self.desc_label.text = "no data found"
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convertir l'image pour Kivy
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
            texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            self.video.texture = texture

    def on_stop(self):
        self.capture.release()


# Lancer l'application
if __name__ == "__main__":
    data_info = {
        "Adam":"Le nouveau",
        "Arthur":"Le barbu",
        "Benjamin":"Un compétiteur",
        "Brian":"Un blond",
        "Emmanuel":"Un grand fournisseur de photo",
        "JB":"Le J!",
        "JL":"Le J!",
        "Loic":"Le violoniste",
        "Mathieu":"Un Blond",
        "Mickael":"Un grand fournisseur de photo",
        "Oren":"Le barde",
        "Pierre":"Le créateur",
        "Renaud":"Un compétiteur",
        "Thi":"La créatrice"
    }
    # Initialisation des modèles et base de données
    facemodel = vgg_face_blank()
    data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
    l = data['layers']
    description = data['meta'][0,0].classes[0,0].description

    copy_mat_to_keras(facemodel)
    featuremodel = Model(inputs=facemodel.layers[0].input, outputs=facemodel.layers[-2].output)

    # featuremodel = get_feature_model()
    # db = generate_database()
    database_path = "database.pkl"
    db = load_database(database_path)
    # Si la base de données est vide, la générer
    if not db:
        db = generate_database()
        # Sauvegarder la base de données générée
        save_database(db, database_path)
    MainApp().run()
