import os
import cv2

import zipfile

import numpy as np

from PIL import Image

def get_image_data() :
    paths = [os.path.join("../Database/yalefaces/train",f)for f in os.listdir(path="../Database/yalefaces/train")]
    # faces will contain the px of the images
    # path will contain the path of the images

    faces = []
    ids = []
    for path in paths :
        #image processing
        #input image
        image = Image.open(path).convert('L')
        image_np = np.array(image,'uint8')
        id = int(os.path.split(path)[1].split(".")[0].replace("subject"," "))
        ids.append(id)
        faces.append(image_np)
    
    return np.array(ids),faces
ids , faces = get_image_data()
# Initialize LBPH (Local Binary Pattern Histogram(lbp))  face recognizer object using OpenCV's computer vision library
lbph_classifier = cv2.faces.LBPHFaceRecognizer_create()

# Train the LBPH classifier using the provided face images (faces) and corresponding labels (ids)
lbph_classifier.train(faces,ids)

# Below line will store the histograms for each one of the images
lbph_classifier.write('lbph_classifier.yml')

#Utilizing the trained model for real-time facial recognition on a test image in the Python environment.

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read("lbph_classifier.yml")

#Now we will check the performance of the model by predicting on a test image

test_image = "../Database/yalefaces/test/subject03.leftlight.gif"
image = Image.open(test_image).convert('L')
image_np = np.array(image,'uint8')

# Before giving the image to the model, let's visualize it first
cv2.imshow(image_np)
# Using the trained model to predict identity of the person in the test image
predictions = lbph_face_classifier.predict(image_np)
print(predictions)

# Retrieving the expected output (ground truth) from the test image file name
expected_output = int(os.path.split(test_image)[1].split('.')[0].replace("subject"," "))
print(expected_output)

# Displaying the predicted(face detection) and expected outputs for comparison
cv2.putText(image_np, 'Pred.' +str(predictions[0]),(10,30),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
cv2.putText(image_np, 'Expec.' +str(expected_output),(10,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0)
)
cv2.imshow(image_np)