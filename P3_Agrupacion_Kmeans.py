from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv("PokemonModSTD.csv")
#df = pd.read_csv("PokemonMod.csv")

X = df.iloc[:,:-1]
Y = pd.DataFrame(df.iloc[:,-1])

X = X.values

clusters = 3

kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_  # Cluster al que pertenece cada registro
centroids = kmeans.cluster_centers_  # Centroide de cada cluster

print("Centroides:", centroids)
print("\nAsignacion:", labels)

# Silhouette
# En terminos generales:
#   Valores negativos representan una Mala separación
#   Valores cercanos a uno son una buena separación
#   Valores cercanos a cero son una separación pobre
from sklearn.metrics import silhouette_score
valor_medido = silhouette_score(X, labels)
print("Coeficiente de Silhouette:", valor_medido)

df["cluster"] = labels
df.to_csv("Pokemon-con-clusters.csv",index=False)



