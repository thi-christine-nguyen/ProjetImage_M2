import os
import cv2
from crop_faces import extract_faces

if __name__ == "__main__":
    directoryIn = "../Database/FaceDataBase"
    directoryOut = "../Database/FaceExtracted"
    try:
        os.mkdir(directoryOut)
        print(f"Directory '{directoryOut}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directoryOut}' already exists.")
        
    for dirname in os.listdir(directoryIn):
        d = os.path.join(directoryIn, dirname)
        dout = os.path.join(directoryOut, dirname)
        # checking if it is a file
        if os.path.isdir(d):
            try:
                os.mkdir(dout)
                print(f"Directory '{dout}' created successfully.")
            except FileExistsError:
                print(f"Directory '{dout}' already exists.")
            for filename in os.listdir(d):
                f = os.path.join(d, filename)
                if os.path.isfile(f):
                    outfile_name = ""
                    try:
                        image = cv2.imread(f)
                        faces = extract_faces(image)
                        outfile_name = os.path.join(os.path.join(directoryOut,dirname), filename)
                        cv2.imwrite(outfile_name, faces[0])
                    except:
                        print(outfile_name)
                        print("is unwriteable")