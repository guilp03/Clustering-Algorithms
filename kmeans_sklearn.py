import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import dataset as dst
from yellowbrick.cluster import KElbowVisualizer

X_principal = dst.train_normalized
X_test_principal = dst.test_normalized
# PCA
#pca = PCA(n_components=2) 
#pca_test = PCA(n_components=2)
#X_principal = pca.fit_transform(X) 
#X_principal = pd.DataFrame(X_principal, columns=['P1', 'P2'])
#X_test_principal = pca_test.fit_transform(dst.test_normalized)
#X_test_principal = pd.DataFrame(X_test_principal, columns=['P1', 'P2'])


model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
visualizer.fit(X_principal)   
visualizer.poof()    

n = 2

kmeans = KMeans(n_clusters=n)
fit_kmeans = kmeans.fit(X_principal)
labels = kmeans.predict(X_test_principal)

silhouette = silhouette_score(X_test_principal, labels)
    
        
print("Best Silhouette Score:", silhouette)
print("Best n_clusters:", n)

def plot_pca(X, model_kmeans=None, print_centroids=False):
    pca = PCA(n_components=2, random_state=33)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title('PCA Analysis')

    if model_kmeans is not None:
        if print_centroids:
            cluster_centers_principal_components = pca.transform(model_kmeans.cluster_centers_)
            num_clusters = cluster_centers_principal_components.shape[0]

            X_clusters = model_kmeans.predict(X)

            for cluster in range(num_clusters):
                indexes = np.where(X_clusters == cluster)
                ax.scatter(X_pca[indexes, 0], X_pca[indexes, 1], s=20, label=f'Cluster #{cluster}')

            ax.scatter(cluster_centers_principal_components[:, 0], cluster_centers_principal_components[:, 1], c='black', s=100, marker='x', label='Centroids')
        else:
            labels = model_kmeans.predict(X)
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=20, label='Clusters')

    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=20)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()

    plt.show()

plot_pca(X_test_principal, kmeans, print_centroids=True)