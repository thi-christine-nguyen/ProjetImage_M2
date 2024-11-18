import numpy as np
from PIL import Image

def calculate_rmse(image1, image2):
    """
    Calcule la racine carrée de l'erreur quadratique moyenne (RMSE) entre deux images.
    Retourne la différence quadratique moyenne (erreur) entre les images.
    """
    # Convertir les images en matrices numpy
    array1 = np.array(image1)
    array2 = np.array(image2)
    
    # Vérifier que les dimensions des images sont identiques
    if array1.shape != array2.shape:
        raise ValueError("Les images doivent avoir la même taille pour être comparées.")
    
    # Calculer la différence quadratique entre les pixels
    diff = np.square(array1 - array2)
    
    # Calculer la moyenne de l'erreur quadratique
    mean_diff = np.sqrt(np.mean(diff))
    
    return mean_diff

def score_to_percentage_rmse(score, image):
    """
    Convertit un score RMSE en pourcentage de similitude.
    Le score RMSE est converti en pourcentage de similitude en fonction de la taille de l'image.
    """
    max_diff_possible = 255 * np.sqrt(image.size[0] * image.size[1] * 3)  # Calcul basé sur la taille de l'image
    similarity_percentage = 100 - (score / max_diff_possible * 100)
    return similarity_percentage

# Fonction principale pour comparer deux images avec l'image de référence
def compare_images(image1, image2, reference):
    """
    Compare les images 1 et 2 avec une image de référence, et renvoie la similitude de chaque image.
    """
    # Calcul du RMSE pour l'image 1 et l'image de référence
    similarity1 = calculate_rmse(image1, reference)
    similarity1_percentage = score_to_percentage_rmse(similarity1, reference)
    
    # Calcul du RMSE pour l'image 2 et l'image de référence
    similarity2 = calculate_rmse(image2, reference)
    similarity2_percentage = score_to_percentage_rmse(similarity2, reference)
    
    # Affichage des résultats
    print(f"Similitude Image 1 - Référence : {similarity1_percentage:.2f}%")
    print(f"Similitude Image 2 - Référence : {similarity2_percentage:.2f}%")
    
    # Choisir l'image la plus similaire
    if similarity1_percentage > similarity2_percentage:
        print("L'image 1 est la plus similaire à l'image de référence.")
    else:
        print("L'image 2 est la plus similaire à l'image de référence.")

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les images (remplacez les chemins par vos propres images)
    image1 = Image.open("CroppedImagesDlib/Thi_24.jpg")
    image2 = Image.open("CroppedImagesHaar/Thi_24.jpg")
    reference = Image.open("Cut_Thi_24.jpg")
    
    # Comparer les images
    compare_images(image1, image2, reference)
