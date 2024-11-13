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
   plt.show()

faces = fetch_olivetti_faces()
images = faces.images
print(images.shape)

features = faces.data  # features
targets = faces.target # targets
#showImgs(images, 100, range(100))

query_img1 = images[71]
print('the target of this face is :', targets[71])
# showImgs([query_img1], 1, [0])
# showImgs(query_img1, 10, range(10))

query_lambda1 = query_img1.reshape(-1)
r = np.array([I.reshape(-1) for I in images])
tot_r = np.zeros(4096)
for ri in r:
   tot_r += ri
   g = (1/400)* tot_r
#showImgs([g.reshape(64,64)], 1, [0])

phi1 = query_lambda1 - g
phi = np.array([I - g for I in r])
print("phi.shape", phi.shape)

# Using the PCA algorithm
pca = PCA(svd_solver='full')

pca.fit(faces["data"])

print("pca.components_.shape", pca.components_.shape)
print("images.shape:", images.shape)

# print(coord.shape) #pourquoi ça ne marche pas

# These lines just for showing the images in (64, 64)format
best_eigenfaces = []
for eigenface in pca.components_[0 : 42]:
   best_eigenfaces.append(eigenface.reshape(64, 64))
showImgs(best_eigenfaces, 42, range(42))

best_eigenfaces = pca.components_[0 : 40]
print("best_eigenfaces.shape : ", best_eigenfaces.shape)

# plt.show()