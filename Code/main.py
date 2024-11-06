import numpy as np
import cv2

def auto_crop_image(image):
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

        # Dessiner les rectangles
        for (x, y, w, h) in fused_faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Image avec rectangles autour des visages", image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

image_path = "images/beatles.jpg"
image = cv2.imread(image_path)

cropped_face = auto_crop_image(image)
