import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import dataset as dst

class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        X_array = X.values
        
        np.random.seed(0)
        self.centroids = X_array[np.random.choice(X_array.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            distances = np.sqrt(((X_array - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            new_centroids = np.array([X_array[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids


    def predict(self, X):
        X_array = X.values

        distances = np.sqrt(((X_array - self.centroids[:, np.newaxis])**2).sum(axis=2))

        labels = np.argmin(distances, axis=0)

        return labels


X_principal = dst.train_normalized
X_test_principal = dst.test_normalized

n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
silhouette_score_list = []
best_score = -1
best_n = None

for n in n_clusters:
    kmeans = KMeans(n_clusters=n)
    fit_kmeans = kmeans.fit(X_principal)
    labels = kmeans.predict(X_test_principal) 
    if n > 1:
        silhouette_avg = silhouette_score(X_test_principal, labels)
        silhouette_score_list.append(silhouette_avg)
    
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n = n

fig, ax = plt.subplots(figsize=(12, 6))

ax.set_xlabel('Número de clusters (K)')
ax.set_ylabel('Pontuação de Silhueta')
ax.set_xticks(n_clusters)
ax.set_title('Pontuação de Silhueta em relação ao Número de clusters')

ax.plot(n_clusters, silhouette_score_list, marker='o', linestyle='-')

plt.show()

print("Best Silhouette Score:", best_score)
print("Best n_clusters:", best_n)

kmeans = KMeans(n_clusters=best_n)
fit_kmeans = kmeans.fit(X_principal)
labels = kmeans.predict(X_test_principal)

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

            for cluster in range(num_clusters):
                indexes = np.where(X_clusters == cluster)
                ax.scatter(X_pca[indexes, 0], X_pca[indexes, 1], s=20, label=f'Cluster #{cluster}')

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

plot_pca(X_test_principal, kmeans, print_centroids=False)