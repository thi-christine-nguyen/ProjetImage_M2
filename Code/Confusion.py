import numpy as np
import matplotlib.pyplot as plt

# Définir les valeurs pour chaque seuil
seuils = [0.1, 0.2, 0.3, 0.4, 0.5]
VP = [5, 19, 23, 23, 23]
FP = [0, 0, 2, 4, 5]
FN = [19, 5, 0, 0, 0]
VN = [4, 4, 3, 1, 0]

# Calculer la précision, le rappel et le F1-score pour chaque seuil
def calculate_precision(VP, FP):
    return VP / (VP + FP) if (VP + FP) != 0 else 0

def calculate_recall(VP, FN):
    return VP / (VP + FN) if (VP + FN) != 0 else 0

def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Initialisation des listes de scores
precisions = [calculate_precision(vp, fp) for vp, fp in zip(VP, FP)]
recalls = [calculate_recall(vp, fn) for vp, fn in zip(VP, FN)]
f1_scores = [calculate_f1(precision, recall) for precision, recall in zip(precisions, recalls)]

# Afficher une matrice de confusion et les métriques en texte pour chaque seuil
for i, seuil in enumerate(seuils):
    # Créer une figure avec deux axes : un pour la matrice, un pour le texte
    fig = plt.figure(figsize=(6, 8))  # Augmenter la hauteur pour inclure le texte
    ax_matrix = fig.add_axes([0.1, 0.4, 0.8, 0.5])  # Axe pour la matrice
    ax_text = fig.add_axes([0.1, 0.1, 0.8, 0.2])  # Axe pour le texte

    # Affichage de la matrice de confusion
    cm = np.array([[VP[i], FP[i]], [FN[i], VN[i]]])
    ax_matrix.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_matrix.set_title(f"Matrice de Confusion (Seuil {seuil})")
    ax_matrix.set_xticks([0, 1])
    ax_matrix.set_yticks([0, 1])
    ax_matrix.set_xticklabels(['Prédit Positif', 'Prédit Négatif'])
    ax_matrix.set_yticklabels(['Réel Positif', 'Réel Négatif'])
    for j in range(2):
        for k in range(2):
            ax_matrix.text(k, j, cm[j, k], ha='center', va='center', color='black')

    # Affichage des scores dans l'axe dédié
    ax_text.axis('off')  # Pas besoin d'axes visibles ici
    scores_text = (
        f"Précision: {precisions[i]:.2f}\n"
        f"Rappel: {recalls[i]:.2f}\n"
        f"F1-Score: {f1_scores[i]:.2f}"
    )
    ax_text.text(0.5, 0.5, scores_text, ha="center", va="center", fontsize=12, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    # Afficher la fenêtre
    plt.show()  # Attendre que l'utilisateur ferme la fenêtre avant de continuer
