import cv2
import os
import numpy as np

def detect_faces(image : cv2.typing.MatLike) -> list[tuple]:
    """
    Detection des visages à l'aide d'Haarcascade
    """
    if image is not None:
        im = image.copy()
       
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(30, 30)
        )

        # Fusionner les rectangles
        fused_faces = []
        for (x, y, w, h) in faces:
            merged = False
            for i, (fx, fy, fw, fh) in enumerate(fused_faces):
                # Vérifie si les rectangles se chevauchent
                if (x < fx + fw and x + w > fx and y < fy + fh and y + h > fy):
                    # Calculer la position et dimension du rectangle fusionné
                    nx = min(x, fx)
                    ny = min(y, fy)
                    nw = max(x + w, fx + fw) - nx
                    nh = max(y + h, fy + fh) - ny
                    fused_faces[i] = (nx, ny, nw, nh)
                    merged = True
                    break
            if not merged:
                fused_faces.append((x, y, w, h))
        return fused_faces


def extract_faces(image : cv2.typing.MatLike) -> list[cv2.typing.MatLike]:
    if image is not None:
        faces_rect = detect_faces(image)
        faces = []
        for (x, y, w, h) in faces_rect:
            faces.append(image[y:y+h, x:x+w])
        return faces

def show_faces(image : cv2.typing.MatLike):
    if image is not None:
        faces_rect = detect_faces(image)
        # Dessiner les rectangles
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Image avec rectangles autour des visages", image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

if __name__ == "__main__":


        
    image_path = "images/beatles.jpg"
    image = cv2.imread(image_path)
    show_faces(image)
    
    # faces = extract_faces(image)
    # for i in range(len(faces)):
    #     cv2.imwrite("images/beatles/"+str(i)+".jpg", faces[i])