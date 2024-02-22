from sklearn.cluster import DBSCAN
import dataset as dst
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA 
import pandas as pd
import matplotlib.pyplot as plt

X = dst.train_normalized

eps_values = [0.1,0.2, 0.3, 0.4, 0.5]
min_samples_values = [20, 25, 30, 35, 40, 45, 50]

best_score = -1
best_eps = None
best_min_samples = None
silhouette_avg = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        db_fit = dbscan.fit(X)
        labels = db_fit.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            silhouette_avg = silhouette_score(X, labels)
        
        if silhouette_avg:
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_eps = eps
                best_min_samples = min_samples

print("Melhor pontuação Silhouette:", best_score)
print("Melhor valor de eps:", best_eps)
print("Melhor valor de min_samples:", best_min_samples)

dbscan = DBSCAN(eps=best_eps, min_samples=best_eps)
db_fit = dbscan.fit(X)
labels = db_fit.labels_

def plot_pca(X, model_dbscan=None):
    pca = PCA(n_components=2, random_state=33)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('PCA Analysis')

    if model_dbscan is not None:
        labels = model_dbscan.labels_
        unique_labels = np.unique(labels)
        core_samples_mask = np.zeros_like(model_dbscan.labels_, dtype=bool)
        core_samples_mask[model_dbscan.core_sample_indices_] = True
        
        for label in unique_labels:
            if label == -1:
                ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], c='black', s=20, label='Noise')
            else:
                ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], s=20, label=f'Cluster #{label}')

        ax.legend()
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=20)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    plt.show()


plot_pca(X, dbscan)
