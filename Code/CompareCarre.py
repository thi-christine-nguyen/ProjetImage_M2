import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_images(image1_path, image2_path):
    """
    Compare deux images et retourne des mesures de similarité et différence.

    Arguments :
    - image1_path : chemin de la première image.
    - image2_path : chemin de la deuxième image.

    Retourne :
    - un dictionnaire avec les valeurs de SSIM et MAE.
    """
    # Charger les images en niveaux de gris
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Vérifier que les deux images ont les mêmes dimensions
    if img1.shape != img2.shape:
        raise ValueError("Les deux images doivent avoir les mêmes dimensions pour être comparées.")

    # Calcul de la similarité structurelle (SSIM)
    ssim_value, _ = ssim(img1, img2, full=True)

    # Calcul de la différence moyenne absolue (MAE)
    mae_value = np.mean(np.abs(img1.astype("float32") - img2.astype("float32")))

    return {
        "SSIM": ssim_value,
        "MAE": mae_value
    }

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez les chemins ci-dessous par vos propres images
    image1_path = "Cut_Thi_25.jpg"
    image2_path = "CroppedImagesHaar/Thi_25.jpg"
    
    try:
        result = compare_images(image1_path, image2_path)
        print(f"SSIM (similarité structurelle) : {result['SSIM']:.4f}")
        print(f"MAE (différence moyenne absolue) : {result['MAE']:.4f}")
    except ValueError as e:
        print(f"Erreur : {e}")
