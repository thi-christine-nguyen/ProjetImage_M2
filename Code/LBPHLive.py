import cv2
import numpy as np
import os
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class DlibCutFaces:
    def __init__(self, image, x1, y1, x2, y2, landmarks) -> None:
        self.image = image
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.landmarks = landmarks

    def draw_face(self)->None:
        cv2.rectangle(self.image, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)
        for i in range(0, 68):  # 68 points de landmarks
            x_landmark = self.landmarks.part(i).x
            y_landmark = self.landmarks.part(i).y
            cv2.circle(self.image, (x_landmark, y_landmark), 1, (255, 0, 0), -1)



def preprocess_image(img):
    return cv2.equalizeHist(img)

def dlib_cut(image):
    if image is not None:
        list_faces = []
        # Convertir en niveaux de gris (requis pour Dlib)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Détection des visages
        faces = detector(gray)
        
        if len(faces) > 0:
            # Dessiner un rectangle autour des visages et afficher les landmarks
            for face in faces:
                landmarks = predictor(gray, face)
                list_faces.append(DlibCutFaces(image, face.left(), face.top(), face.right(), face.bottom(), landmarks))
            
            # Retourner l'image découpée, l'image annotée et les dimensions du visage
            # print(f"Found {len(faces)} faces!")
            return list_faces
    
    # Si aucun visage n'est détecté, retourner l'image originale
    return []

def main(lbph_classifier, list_class):
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    ready_to_detect_identity = True
    name = ""
    frame_counter = 0
    
    while vc.isOpened():
        _, frame = vc.read()
        list_faces : list[DlibCutFaces] = dlib_cut(frame)
        for face in list_faces:
            face.draw_face()
            if face.x1<0 or face.x2 <0 or face.y1<0 or face.y2<0:
                continue
            img = frame.copy()[face.x1:face.x2, face.y1:face.y2]
            if img.shape[0] == 0 or img.shape[1]==0:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = preprocess_image(img)
            label, confidence = lbph_classifier.predict(img)
            cv2.putText(img = frame, text = list_class[label], org = (int(face.x1),int(face.y2+30)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, thickness=2, fontScale=1, color=(0, 255, 0))
        key = cv2.waitKey(1)
        cv2.imshow("preview", frame)

        if key == 27: # quitter avec la touche ESC
            break
    
    cv2.destroyWindow("preview")

if __name__ == "__main__":
    list_class = os.listdir("../Database/FaceExtracted")

    faces = []
    ids = []
    for i in range(0,len(list_class)):
        image_path = f"../Database/FaceExtracted/{list_class[i]}/"
        list_img = os.listdir(image_path)
        for j in range(3):
            if os.path.exists(image_path + list_img[0]):
                img = cv2.imread(image_path + list_img[0], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # img = preprocess_image(img)
                    faces.append(img)
                    ids.append(i)
    ids = np.array(ids, dtype='int32')
    lbph_classifier = cv2.face.LBPHFaceRecognizer_create()

    # Entraîner le modèle
    lbph_classifier.train(faces, ids)

    main(lbph_classifier, list_class)

    
