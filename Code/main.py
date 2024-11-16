# -*- coding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Flatten, Dropout, Activation, Permute
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
K.set_image_data_format( 'channels_last' )

import numpy as np
import cv2
from scipy.spatial.distance import cosine as dcos
from scipy.io import loadmat

import os
from multiprocessing.dummy import Pool
#Bonus :
from threading import Thread
import win32com.client 

import dlib

detector = dlib.get_frontal_face_detector()

# Découpage d'image avec dlib
def auto_crop_image(image):
    if image is not None:
        im = image.copy()
        
        # Convertir en niveaux de gris (requis pour Dlib)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Détection des visages
        faces = detector(gray)
        
        if len(faces) > 0:
            # Dessiner un rectangle autour des visages
            for face in faces:
                x, y, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
            
            # Prendre le premier visage détecté
            first_face = faces[0]
            x, y, x2, y2 = first_face.left(), first_face.top(), first_face.right(), first_face.bottom()
            w, h = x2 - x, y2 - y
            
            # Calculer les dimensions du cadre centré autour du visage
            center_x = x + w / 2
            center_y = y + h / 2
            height, width, channels = im.shape
            b_dim = min(max(w, h) * 1.2, width, height)  # Ajustement pour inclure un peu de contexte
            box = [
                int(center_x - b_dim / 2),
                int(center_y - b_dim / 2),
                int(center_x + b_dim / 2),
                int(center_y + b_dim / 2)
            ]
            
            # Vérifier que le cadre reste dans les limites de l'image
            if box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height:
                crpim = im[box[1]:box[3], box[0]:box[2]]
                # Redimensionner à 224x224 pour un traitement uniforme
                crpim = cv2.resize(crpim, (224, 224), interpolation=cv2.INTER_AREA)
                print(f"Found {len(faces)} faces!")
                return crpim, image, (x, y, w, h)
    return None, image, (0, 0, 0, 0)

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


def generate_database(folder_img="FaceDataBase", save_cropped_images=True, save_folder="CroppedImages"):
    database = {}
    
    # Créer le dossier pour enregistrer les images découpées si nécessaire
    if save_cropped_images and not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Parcours des sous-dossiers (chaque sous-dossier représente une personne)
    for person_folder in os.listdir(folder_img):
        person_folder_path = os.path.join(folder_img, person_folder)
        
        if os.path.isdir(person_folder_path):
            person_images = []
            
            # Parcours de toutes les images dans le dossier de la personne
            for img_file in os.listdir(person_folder_path):
                img_path = os.path.join(person_folder_path, img_file)
                
                try:
                    if os.path.isfile(img_path) and img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img = cv2.imread(img_path)
                        
                        # Recadrage automatique de l'image
                        crpim, srcimg, (x, y, w, h) = auto_crop_image(img)
                        
                        if crpim is not None:
                            # Enregistrer l'image recadrée si nécessaire
                            if save_cropped_images:
                                cropped_img_path = os.path.join(save_folder, f"{person_folder}_{img_file}")
                                cv2.imwrite(cropped_img_path, crpim)
                            
                            # Extraction des caractéristiques de l'image avec le modèle
                            vector_image = crpim[None, ...]
                            person_vector = featuremodel.predict(vector_image)[0, :]
                            
                            # Ajout du vecteur à la base de données
                            if person_folder not in database:
                                database[person_folder] = []
                            database[person_folder].append(person_vector)
                except Exception as e:
                    print(f"Erreur avec l'image {img_path}: {e}")
    
    return database


# Fonction pour trouver la personne la plus proche en utilisant la base de données
def find_closest(img, database, min_detection=2.5):
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
    
    if dmin > min_detection:
        umin = ""
    
    return umin, dmin


def main(database):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    ready_to_detect_identity = True
    name = ""
    frame_counter = 0
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        frame_counter += 1
        
        # Ne traiter que chaque 10e frame
        if frame_counter % 10 == 0:
            imgcrop, img, (x, y, w, h) = auto_crop_image(frame)
            
            if ready_to_detect_identity and imgcrop is not None:
                # Empêcher une nouvelle détection pendant une identification en cours
                ready_to_detect_identity = False
                
                # Utilisation du pool pour la détection de l'identité
                pool = Pool(processes=1)
                name, ready_to_detect_identity = pool.apply_async(recognize_image, [imgcrop, database]).get()
                pool.close()
                
                # Affichage du nom de la personne identifiée
                cv2.putText(img = frame, text = name, org = (int(x),int(y+h+20)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, thickness=2, fontScale=1, color=(0, 255, 0))
            
            key = cv2.waitKey(100)
            cv2.imshow("preview", img)

            if key == 27: # quitter avec la touche ESC
                break
    
    cv2.destroyWindow("preview")

def recognize_image(imgcrop, database):
    name, dmin = find_closest(imgcrop, database)
    return name, True

facemodel = vgg_face_blank()
data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
l = data['layers']
description = data['meta'][0,0].classes[0,0].description

copy_mat_to_keras(facemodel)
featuremodel = Model(inputs=facemodel.layers[0].input, outputs=facemodel.layers[-2].output)

db = generate_database()

main(db)