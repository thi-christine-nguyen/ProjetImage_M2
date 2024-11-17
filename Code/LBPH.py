import cv2
import numpy as np
import os

def LBP_ORL(train_sample:int):
    def preprocess_image(img):
        return cv2.equalizeHist(img)

    faces = []
    ids = []
    for s in range(1, 41):
        for i in range(1,train_sample+1):
            image_path = f"../Database/ORL/s{s}/{i}.pgm"
            if os.path.exists(image_path):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = preprocess_image(img)
                    faces.append(img)
                    ids.append(s)
    ids = np.array(ids, dtype='int32')
    
    lbph_classifier = cv2.face.LBPHFaceRecognizer_create()

    # Entraîner le modèle
    lbph_classifier.train(faces, ids)

    t = 0
    f = 0
    for s in range(1, 41):
        for i in range(train_sample+1, 11):
            image_path = f"../Database/ORL/s{s}/{i}.pgm"
            if os.path.exists(image_path):
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = preprocess_image(img)
                    label, confidence = lbph_classifier.predict(img)
                    t += int(label == s)
                    f += int(label != s)
                    
    return (t,f)

if __name__ == "__main__":
    with open("log.txt", "w") as log_file:
        for i in range(1, 10):
            true_pred, false_pred = LBP_ORL(10 - i)
            total = true_pred + false_pred
            accuracy = (true_pred / total * 100) if total > 0 else 0
            log_file.write(
                f"Train with {(10 - i) * 40} samples, {i} sample per class, "
                f"{true_pred} true, {false_pred} false for {total} predictions "
                f"({accuracy:.2f}% accuracy)\n"
            )