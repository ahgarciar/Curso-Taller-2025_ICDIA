from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv("PokemonModSTD.csv")
#df = pd.read_csv("PokemonMod.csv")

X = df.iloc[:,:-1]
Y = pd.DataFrame(df.iloc[:,-1])

X = X.values

clusters = 3

# Aplicar K-Means
kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Centroides finales:\n", centroids)
print("\nAsignación de Clusters:\n", labels)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Mostrar los componentes principales
print("Componentes principales:\n", pca.components_)
#print("Varianza explicada por componente:", pca.explained_variance_ratio_)
#print("Varianza explicada total:", sum(pca.explained_variance_ratio_))

centroids_pca = pca.transform(centroids)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroides')
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clusters en 2D - PCA")
plt.legend()
plt.show()