# -*- coding: utf-8 -*-
from keras.models import load_model
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

import time
import pickle
import dlib

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
            # Dessiner un rectangle autour des visages et afficher les landmarks
            for face in faces:
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
            
            # Calculer les dimensions du cadre centré autour du visage
            center_x = x + w / 2
            center_y = y + h / 2
            height, width, channels = im.shape
            
            # Ajustement pour inclure du contexte autour du visage
            margin = 0.4  # Marge contextuelle (40% des dimensions du visage)
            box_w = w * (1 + margin)
            box_h = h * (1 + margin)
            box_dim = min(box_w, box_h, width, height)  # Limiter la taille à l'image
            
            # Calculer les coordonnées du cadre centré
            x_start = max(0, int(center_x - box_dim / 2))
            y_start = max(0, int(center_y - box_dim / 2))
            x_end = min(width, int(center_x + box_dim / 2))
            y_end = min(height, int(center_y + box_dim / 2))
            
            # Découper et redimensionner l'image
            crpim = im[y_start:y_end, x_start:x_end]
            crpim = cv2.resize(crpim, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Retourner l'image découpée, l'image annotée et les dimensions du visage
            print(f"Found {len(faces)} faces!")
            return crpim, image, (x, y, w, h)
    
    # Si aucun visage n'est détecté, retourner l'image originale
    return None, image, (0, 0, 0, 0)


def haar(image):
    if image is not None:
        # Load HaarCascade from the file with OpenCV only once (outside of the function)
        if not hasattr(haar, 'faceCascade'):
            haar.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Convert to grayscale once
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image using optimized scaleFactor and minNeighbors
        faces = haar.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,  # Reduced minNeighbors for faster detection
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # # Draw rectangle only if faces are detected
            # for (x, y, w, h) in faces:
            #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Only process the first face for cropping and resizing
            (x, y, w, h) = faces[0]
            center_x = x + w // 2
            center_y = y + h // 2
            height, width, _ = image.shape

            # Calculate bounding box with some padding
            b_dim = min(max(w, h) * 1.2, width, height)
            box = [center_x - b_dim // 2, center_y - b_dim // 2, center_x + b_dim // 2, center_y + b_dim // 2]
            box = [int(coord) for coord in box]

            # Crop and resize image if within bounds
            if box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height:
                crpim = image[box[1]:box[3], box[0]:box[2]]
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
    start_time = time.time()
    
    database = {}
    
    # # Créer le dossier pour enregistrer les images découpées si nécessaire
    # if save_cropped_images and not os.path.exists(save_folder):
    #     os.makedirs(save_folder)
    
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
                        crpim, srcimg, (x, y, w, h) = dlib_cut(img)
                        
                        if crpim is not None:
                            # Enregistrer l'image recadrée si nécessaire
                            if save_cropped_images:
                                cropped_img_path = os.path.join(save_folder, f"{person_folder}_{img_file}")
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
    elapsed_time = time.time() - start_time
    print(f"Temps écoulé pour générer la base de données : {elapsed_time:.2f} secondes.")

    # for person, vectors in database.items():
    #     print(f"Nom : {person}")
    #     for i, vector in enumerate(vectors):
    #         print(f"  Vecteur {i + 1} : {vector}")
    
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
    
    if dmin > min_detection:
        umin = "Inconnu"
    
    return umin, dmin

# def find_closest_distance(img, database, min_detection=0.7):
#     imarr1 = np.asarray(img)
#     imarr1 = imarr1[None, ...]  # Ajouter une dimension pour le batch
    
#     # Prédiction du vecteur de caractéristiques de l'image
#     fvec1 = featuremodel.predict(imarr1)[0, :]
    
#     # Normaliser le vecteur de caractéristiques (si vous utilisez la distance euclidienne)
#     fvec1 = fvec1 / np.linalg.norm(fvec1)
    
#     # Recherche de la personne la plus proche dans la base de données
#     dmin = float('inf')
#     umin = ""
    
#     for person, vectors in database.items():
#         for fvec2 in vectors:
#             # Normaliser fvec2
#             fvec2 = fvec2 / np.linalg.norm(fvec2)
            
#             # Calcul de la distance euclidienne entre fvec1 et fvec2
#             dist = np.linalg.norm(fvec1 - fvec2)
            
#             if dist < dmin:
#                 dmin = dist
#                 umin = person
    
#     # Si la distance minimale est supérieure à un seuil, retourner une personne inconnue
#     if dmin > min_detection:
#         print("haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
#         umin = "Inconnu"
    
#     return umin, dmin


def recognize_image(imgcrop, database):
    name, dmin = find_closest_angle(imgcrop, database)
    print(name)
    return name, True


def main(database):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    ready_to_detect_identity = True
    name = ""
    frame_counter = 0
    cpt = 0
    last_detected_position = None  # Pour stocker la dernière position détectée
    current_name = None  # Variable pour garder le nom actuel même si la détection ne change pas
    prev_time = time.time()

    while vc.isOpened():
        _, frame = vc.read()
        cpt += 1

        # Ne traiter qu'une frame sur 10 pour améliorer les performances
        if cpt >= 10:
            cpt = 0
            img = frame

            # Recadrer automatiquement l'image et détecter le visage
            imgcrop, img, (x, y, w, h) = dlib_cut(img)

            if imgcrop is not None:
                # Vérifier si un visage est détecté à proximité du précédent
                if last_detected_position:
                    last_x, last_y, last_w, last_h = last_detected_position
                    movement_threshold = 50  # Seuil en pixels pour considérer une "téléportation"

                    if abs(x - last_x) > movement_threshold or abs(y - last_y) > movement_threshold:
                        print("Mouvement trop important, redétection complète...")
                        last_detected_position = (x, y, w, h)
                    else:
                        print("Visage suivi : Position similaire au précédent.")
                else:
                    last_detected_position = (x, y, w, h)

                # Si prêt à détecter une nouvelle identité
                if ready_to_detect_identity:
                    ready_to_detect_identity = False

                    # Utilisation d'un thread pour l'identification
                    pool = Pool(processes=1)
                    current_name, ready_to_detect_identity = pool.apply_async(recognize_image, [imgcrop, database]).get()
                    pool.close()

                # Si un nom est détecté, garder l'affichage de ce nom
                if current_name:
                    name = current_name  # Mettre à jour le nom actuel
                    cv2.putText(
                        img=frame, text=name, org=(int(x), int(y + h + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2,
                        fontScale=1, color=(0, 255, 0)
                    )
                    # Dessiner un rectangle autour du visage
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                print("Aucun visage détecté sur cette frame.")
                name = ""

        if name:
            cv2.putText(
                img=frame, text=name, org=(int(x), int(y + h + 20)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, thickness=2,
                fontScale=1, color=(0, 255, 0)
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # current_time = time.time()
        # fps = 1 / (current_time - prev_time)
        # prev_time = current_time
        # print(fps)

      
        # Affichage et gestion des événements de la fenêtre
        cv2.imshow("preview", frame)
        key = cv2.waitKey(1)
        if key == 27:  # Quitter avec la touche ESC
            break

    cv2.destroyWindow("preview")



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


main(db)

# Test fonction angle
# # Charger une image
# image_path = "FaceDataBase/Thi/24.jpg"  # Remplacez par le chemin de votre image
# image = cv2.imread(image_path)

# # Vérifier si l'image a été correctement chargée
# if image is None:
#     print("Erreur : Impossible de charger l'image.")
# else:
#     # Appliquer le découpage d'image pour détecter et recadrer le visage
#     cropped_image, annotated_image, face_dimensions = dlib_cut(image)
    
#     if cropped_image is not None:
#         # Charger ou créer votre base de données
#         database = generate_database(folder_img="FaceDataBase")
        
#         # Effectuer la reconnaissance faciale
#         recognized_name, is_recognized = recognize_image(cropped_image, database)
        
#         if is_recognized:
#             # Ajouter le texte sur l'image annotée
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 1
#             color = (255, 0, 0)  # Couleur bleue pour le texte
#             thickness = 2
#             position = (10, 30)  # Position du texte
            
#             # Ajouter le texte (nom reconnu) sur l'image annotée
#             cv2.putText(annotated_image, recognized_name, position, font, font_scale, color, thickness, cv2.LINE_AA)
#             print(f"Visage reconnu : {recognized_name}")
#         else:
#             print("Aucun visage correspondant trouvé dans la base de données.")
        
#         # Afficher l'image annotée avec le texte ajouté
#         cv2.imshow("Image Annotée avec Nom", annotated_image)
#         cv2.waitKey(1)
#         cv2.destroyAllWindows()
#     else:
#         print("Aucun visage détecté dans l'image.")



# def process_test_folder(test_folder, output_folder, database_folder):
#     # Charger ou créer votre base de données
#     database = generate_database(folder_img=database_folder)
    
#     # Dossier pour les images non reconnues
#     unknown_folder = os.path.join(output_folder, "Pas de visage")
#     os.makedirs(unknown_folder, exist_ok=True)
    
#     # Parcourir toutes les images du dossier Test
#     for file_name in os.listdir(test_folder):
#         image_path = os.path.join(test_folder, file_name)
        
#         # Vérifier si le fichier est une image
#         if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#             continue
        
#         image = cv2.imread(image_path)

#         # Vérifier si l'image a été correctement chargée
#         if image is None:
#             print(f"Erreur : Impossible de charger l'image {file_name}.")
#             continue
        
#         # Appliquer le découpage d'image pour détecter et recadrer le visage
#         cropped_image, annotated_image, face_dimensions = dlib_cut(image)
        
#         if cropped_image is not None:
#             # Effectuer la reconnaissance faciale
#             recognized_name, is_recognized = recognize_image(cropped_image, database)
            
#             if is_recognized:
#                 # Créer le dossier pour la personne reconnue s'il n'existe pas
#                 person_folder = os.path.join(output_folder, recognized_name)
#                 os.makedirs(person_folder, exist_ok=True)
                
#                 # Ajouter le texte (nom reconnu) sur l'image annotée
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 font_scale = 1
#                 color = (255, 0, 0)  # Couleur bleue pour le texte
#                 thickness = 2
#                 position = (10, 30)  # Position du texte
#                 cv2.putText(annotated_image, recognized_name, position, font, font_scale, color, thickness, cv2.LINE_AA)
                
#                 # Sauvegarder l'image annotée dans le dossier de la personne reconnue
#                 output_path = os.path.join(person_folder, file_name)
#                 cv2.imwrite(output_path, annotated_image)
                
#                 print(f"Visage reconnu : {recognized_name}, image sauvegardée dans {output_path}")
#             else:
#                 # Enregistrer l'image non reconnue dans le dossier "Autres"
#                 output_path = os.path.join(unknown_folder, file_name)
#                 cv2.imwrite(output_path, image)
#                 print(f"Aucun visage correspondant trouvé pour {file_name}, image sauvegardée dans 'Autres'.")
#         else:
#             # Enregistrer l'image non analysable dans le dossier "Autres"
#             output_path = os.path.join(unknown_folder, file_name)
#             cv2.imwrite(output_path, image)
#             print(f"Aucun visage détecté dans {file_name}, image sauvegardée dans 'Autres'.")

# # Configuration des dossiers
# test_folder = "Test"  # Dossier contenant les images à tester
# output_folder = "RecognizedFaces"  # Dossier de sortie pour les images reconnues
# database_folder = "FaceDataBase_cut"  # Dossier contenant la base de données des visages

# # Appel de la fonction principale
# process_test_folder(test_folder, output_folder, database_folder)