from sklearn.cluster import DBSCAN
import dataset as dst
from sklearn.metrics import silhouette_score
import numpy as np

X = dst.df_scaler
eps_values = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
min_samples_values = [5, 10, 15, 20]

best_score = -1
best_eps = None
best_min_samples = None
silhouette_avg = None

for eps in eps_values:
    for min_samples in min_samples_values:
        # Executar o DBSCAN com os parâmetros atuais
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Calcular a pontuação Silhouette
        if n_clusters >0:
            silhouette_avg = silhouette_score(X, labels)
        
        # Verificar se é a melhor pontuação até agora
        if silhouette_avg:
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_eps = eps
                best_min_samples = min_samples

# Imprimir os melhores parâmetros
print("Melhor pontuação Silhouette:", best_score)
print("Melhor valor de eps:", best_eps)
print("Melhor valor de min_samples:", best_min_samples)


import matplotlib.pyplot as plt

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clusters resultantes do DBSCAN')
plt.colorbar(label='Cluster')
plt.show()

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("Número de clusters identificados:", n_clusters)