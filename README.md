# Moteur de Recommandation de Films par SVD (Singular Value Decomposition)

**Auteur :** Kardigue Diallo  
**Formation :** Licence en Mathématiques & Informatique  
**Spécialité :** Data Science, Machine Learning & Algèbre Appliquée  

---

## Présentation du Projet

Ce projet met en œuvre un **système de recommandation basé sur le filtrage collaboratif** à l'aide de la **Décomposition en Valeurs Singulières (SVD)**. 

L'objectif est d'exploiter les propriétés de l'algèbre linéaire pour prédire les préférences des utilisateurs sur des films non encore vus et réduire la dimensionnalité de données creuses (*sparse data*).

---

## Fondations Mathématiques

Toute matrice de notes $A \in \mathbb{R}^{m \times n}$ (représentant $m$ utilisateurs et $n$ films) peut être décomposée sous la forme :

$$A = U \cdot \Sigma \cdot V^T$$

Où :
* **$U \in \mathbb{R}^{m \times m}$** : Matrice orthogonale (profils latents des utilisateurs).
* **$\Sigma \in \mathbb{R}^{m \times n}$** : Matrice diagonale contenant les **valeurs singulières** $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
* **$V^T \in \mathbb{R}^{n \times n}$** : Matrice orthogonale (caractéristiques latentes des films).

### SVD Tronquée :
En ne conservant que les $k$ plus grandes valeurs singulières ($k \ll \min(m, n)$), nous obtenons la meilleure approximation de rang $k$ de la matrice originale :

$$A_k = U_k \cdot \Sigma_k \cdot V_k^T$$

Cette réduction permet de filtrer le bruit, d'extraire les structures sémantiques sous-jacentes et de reconstruire les notes manquantes.
---

## Technologies Utilisées

* **Python 3**
* **NumPy** : Calcul matriciel optimisé et SVD (`np.linalg.svd`)
* **Pandas** : Manipulation des structures de données

---

## 🚀 Exécution du Projet

```bash
python svd_recommender.py
