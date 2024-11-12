#source https://hassan-id-mansour.medium.com/face-recognition-using-eigenfaces-python-b857b2599ed0

import cv2
import math
import numpy as np
import  pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
plt.style.use('dark_background')

def showImgs(imgs, n_imgs, i_imgs):
   n = sqrt(n_imgs)
   m = n
   p = 1
   if n != int(n):
      n = int(n)
      m = n + 1
   print("n, m", n, m)
   fig = plt.figure()
   for i in i_imgs:
      fig.add_subplot(int(n), int(m), p)
      plt.imshow(imgs[i], cmap='gray')
      plt.axis('off')
      p += 1

faces = fetch_olivetti_faces()
images = faces.images
images.shape

# features = faces.data  # features
# targets = faces.target # targets
# showImgs(images, 100, range(100))

query_img1 = images[71]
# print('the target of this face is :', targets[71])
# showImgs([query_img1], 1, [0])

query_γ1 = query_img1.reshape(-1)
Γ = np.array([I.reshape(-1) for I in images])
Σ_Γi = np.zeros(4096)
for Γi in Γ:
   Σ_Γi += Γi
   Ψ = (1/400)* Σ_Γi
#showImgs([Ψ.reshape(64,64)], 1, [0])

φ1 = query_γ1 - Ψ
Φ = np.array([I - Ψ for I in Γ])
Φ.shape

# Using the PCA algorithm
pca = PCA(svd_solver='full')
pca.fit(faces["data"])
# These lines just for showing the images in (64, 64)format
best_eigenfaces = []
for eigenface in pca.components_[0 : 40]:
   best_eigenfaces.append(eigenface.reshape(64, 64))
showImgs(best_eigenfaces, 40, range(40))

plt.show()