import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image

def crop_and_resize_image(image_path, save_path, target_size=(224, 224)):
    """
    Permet de sélectionner une région carrée dans une image,
    de la redimensionner et de la sauvegarder.
    
    :param image_path: Chemin de l'image source.
    :param save_path: Chemin pour sauvegarder l'image redimensionnée.
    :param target_size: Taille de l'image redimensionnée (par défaut 224x224).
    """
    # Charger l'image
    img = Image.open(image_path)
    
    # Initialiser les coordonnées de sélection
    global rect_coords
    rect_coords = None

    # Fonction de gestion de clics pour capturer les coordonnées
    def on_select(eclick, erelease):
        global rect_coords
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        # Forcer la sélection en carré (1:1 ratio)
        size = min(abs(x2 - x1), abs(y2 - y1))  # Taille carrée maximale
        if x2 > x1:
            x2 = x1 + size
        else:
            x2 = x1 - size
        if y2 > y1:
            y2 = y1 + size
        else:
            y2 = y1 - size
        
        rect_coords = (int(x1), int(y1), int(x2), int(y2))

    # Créer la figure et activer RectangleSelector
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title("Sélectionnez une région carrée, puis fermez la fenêtre.")
    rectangle_selector = RectangleSelector(
        ax,
        on_select,
        drawtype="box",
        useblit=True,
        button=[1],  # Souris gauche uniquement
        interactive=True,
        spancoords='pixels'
    )

    # Afficher l'image pour la sélection
    plt.show()

    # Vérifier que la sélection a été faite
    if rect_coords is None:
        print("Aucune sélection effectuée. Opération annulée.")
        return

    # Découper et redimensionner la région sélectionnée
    cropped_img = img.crop(rect_coords)
    resized_img = cropped_img.resize(target_size, Image.ANTIALIAS)

    # Sauvegarder l'image redimensionnée
    resized_img.save(save_path)
    print(f"L'image redimensionnée a été sauvegardée sous : {save_path}")

# Utilisation du programme
image_path = "FaceDataBase/Pierre/10.jpg"  # Remplacez par le chemin de votre image
save_path = "image_redimensionnee.jpg"  # Chemin pour sauvegarder l'image redimensionnée
crop_and_resize_image(image_path, save_path)
