import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import dataset as dst


X_principal = dst.train_normalized
X_test_principal = dst.test_normalized


n_clusters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class FuzzyCMeans:
    def __init__(self, n_clusters=2, m=2, max_iter=100, error=1e-5):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.centroids = None
    
    def fit(self, X):
        centroids, _, _, _, _, _, _ = fuzz.cluster.cmeans(
        X.T, 
        c=self.n_clusters, 
        m=self.m, 
        error=self.error, 
        maxiter=self.max_iter
    )
        self.centroids = centroids
    
    def predict(self, X):
        X = X.T
        u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        X,
        self.centroids,
        m=self.m,
        error=self.error,
        maxiter=self.max_iter
        )
        labels = np.argmax(u, axis=0)
        return labels



best_score = -1
best_n = None

for clusters in n_clusters:
    # Fitting the Fuzzy C-Means]
    fuzzy = FuzzyCMeans(n_clusters=clusters)
    fuzzy_fit = fuzzy.fit(X_principal)
    labels = fuzzy.predict(X_test_principal)
    
    # Calculate Silhouette score
    if clusters > 1:
        silhouette_avg = silhouette_score(X_test_principal, labels)
        
        # Check if it's the best score so far
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n = clusters

print("Best Silhouette Score:", best_score)
print("Best n_clusters:", best_n)

# Re-fit with the best number of clusters
fuzzy = FuzzyCMeans(n_clusters=best_n)
fuzzy_fit = fuzzy.fit(X_principal)
labels = fuzzy.predict(X_test_principal)

def plot_pca(X, model=None, print_centroids=False):
    pca = PCA(n_components=2, random_state=33)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('PCA Analysis')

    if model is not None:
        if print_centroids:
            cluster_centers_principal_components = pca.transform(model.cluster_centers_)
            num_clusters = cluster_centers_principal_components.shape[0]

            X_clusters = model.predict(X)

            # For each cluster, plot their respective X data instances
            for cluster in range(num_clusters):
                indexes = np.where(X_clusters == cluster)
                ax.scatter(X_pca[indexes, 0], X_pca[indexes, 1], s=20, label=f'Cluster #{cluster}')

            # For each cluster centroid, plot the centroid
            ax.scatter(cluster_centers_principal_components[:, 0], cluster_centers_principal_components[:, 1], c='black', s=100, marker='x', label='Centroids')
        else:
            labels = model.predict(X)
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=20, label='Clusters')

    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=20)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()

    plt.show()

plot_pca(X_test_principal, fuzzy, print_centroids=False)
