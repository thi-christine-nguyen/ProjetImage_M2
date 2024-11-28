import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score

# Données (seuils et valeurs de VP, FP, FN, VN)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
vp = [5, 19, 23, 23, 23]
fp = [0, 0, 2, 4, 5]
fn = [19, 5, 0, 0, 0]
vn = [4, 4, 3, 1, 0]

# Calcul des différentes métriques à chaque seuil
f1_scores = []
tpr = []  # Taux de vrais positifs (recall)
fpr = []  # Taux de faux positifs
precision = []

for i in range(len(thresholds)):
    # Précision, rappel, F1-score
    p = vp[i] / (vp[i] + fp[i]) if (vp[i] + fp[i]) > 0 else 0
    r = vp[i] / (vp[i] + fn[i]) if (vp[i] + fn[i]) > 0 else 0
    f1 = f1_score([1] * (vp[i] + fn[i]) + [0] * (fp[i] + vn[i]),
                  [1] * vp[i] + [0] * fp[i] + [1] * fn[i] + [0] * vn[i])
    
    # Calcul de FPR et TPR
    tpr.append(r)  # TPR = rappel
    fpr.append(fp[i] / (fp[i] + vn[i]) if (fp[i] + vn[i]) > 0 else 0)  # FPR
    
    # Ajouter le F1 score à la liste
    f1_scores.append(f1)

# Générer la courbe ROC
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='b', label=f'Courbe ROC (AUC = {auc(fpr, tpr):.2f})')

# Affichage de la ligne "Random Guess"
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label="Random Guess")

# Ajouter les seuils à la courbe
for i, threshold in enumerate(thresholds):
    plt.text(fpr[i], tpr[i], f'{threshold}', fontsize=12, color='black', ha='right', va='bottom')

plt.xlabel('Taux de faux positifs (FPR)')
plt.ylabel('Taux de vrais positifs (TPR)')
plt.title('Courbe ROC avec Seuils')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
