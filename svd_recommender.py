import numpy as np
import pandas as pd

class SVDRecommender:
    """
    Système de recommandation basé sur la Décomposition en Valeurs Singulières (SVD).
    Développé par Kardigue Diallo - Licencié en Mathématiques & Informatique.
    """
    def __init__(self, n_factors=2):
        """
        :param n_factors: Nombre de facteurs latents (valeurs singulières à conserver)
        """
        self.n_factors = n_factors
        self.U = None
        self.S = None
        self.Vt = None
        self.user_item_matrix = None
        
    def fit(self, user_item_matrix):
        """
        Calcule la SVD de la matrice utilisateurs-films.
        :param user_item_matrix: DataFrame Pandas (Lignes = Utilisateurs, Colonnes = Films)
        """
        self.user_item_matrix = user_item_matrix
        # 1. Centrage des données par utilisateur (pour éliminer les biais individuels)
        self.user_means = user_item_matrix.mean(axis=1)
        matrix_centered = user_item_matrix.sub(self.user_means, axis=0).fillna(0).values
        
        # 2. Décomposition SVD via NumPy
        U, S, Vt = np.linalg.svd(matrix_centered, full_matrices=False)
        
        # 3. Troncature : conservation des k premiers facteurs latents
        self.U = U[:, :self.n_factors]
        self.S = np.diag(S[:self.n_factors])
        self.Vt = Vt[:self.n_factors, :]
        
        print(f"SVD calculée avec succès. Facteurs latents conservés : {self.n_factors}")

    def predict_ratings(self):
        """
        Reconstruit la matrice de notes prédites (U_k * Sigma_k * V_k^T + Moyennes).
        """
        predicted_centered = np.dot(np.dot(self.U, self.S), self.Vt)
        predicted_ratings = pd.DataFrame(
            predicted_centered, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.columns
        )
        return predicted_ratings.add(self.user_means, axis=0)

    def recommend(self, user_id, top_n=3):
        """
        Recommande les top_n films non vus pour un utilisateur donné.
        """
        predictions = self.predict_ratings()
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Filtrer uniquement les films que l'utilisateur n'a pas encore notés
        unseen_movies = user_ratings[user_ratings.isna()].index
        user_predictions = predictions.loc[user_id, unseen_movies]
        
        # Trier et retourner les meilleures recommandations
        return user_predictions.sort_values(ascending=False).head(top_n)


# --- EXEMPLE D'EXÉCUTION ET DÉMONSTRATION ---
if __name__ == "__main__":
    # Matrice de notes simulée (5 Utilisateurs x 5 Films)
    # NaN représente un film non encore vu
    data = {
        'Inception (Sci-Fi)': [5, 4, np.nan, 1, 2],
        'Interstellar (Sci-Fi)': [5, 5, 1, np.nan, 1],
        'Titanic (Romance)': [1, np.nan, 5, 4, 5],
        'The Notebook (Romance)': [np.nan, 1, 4, 5, 4],
        'Avengers (Action)': [4, 5, 2, 1, np.nan]
    }
    users = ['Utilisateur_1', 'Utilisateur_2', 'Utilisateur_3', 'Utilisateur_4', 'Utilisateur_5']
    
    df_ratings = pd.DataFrame(data, index=users)
    print("--- Matrice de notes originale (avec valeurs manquantes) ---")
    print(df_ratings)
    print("\n" + "="*60 + "\n")

    # Initialisation et entraînement du modèle
    recommender = SVDRecommender(n_factors=2)
    recommender.fit(df_ratings)

    # Recommandation pour l'utilisateur 1
    recom_u1 = recommender.recommend('Utilisateur_1', top_n=2)
    print("\n--- Recommandations personnalisées pour Utilisateur_1 ---")
    print(recom_u1)
