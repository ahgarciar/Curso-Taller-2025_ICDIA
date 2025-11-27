from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math as m

df = pd.read_csv("PokemonModSTD.csv")
# df = pd.read_csv("PokemonMod.csv")

X = df.iloc[:, :-1].values

# Prueba de diferentes valores de k
wcss = []

kMax = int(m.sqrt(len(X))) + 1
k_values = range(1, kMax)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    print(f"K={k}  WCSS={kmeans.inertia_}")

# Busqueda del codo (valor de k) donde la caída del WCSS sea mayor
deltas = np.diff(wcss) # diferencia absoluta entre puntos consecutivos
# el codo esta donde hay mayor cambio negativo
best_k = np.argmax(-deltas) + 1   # +1 para tomar en cuenta el desfase por diff

print(f"\nMejor valor de k detectado: {best_k}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', linestyle='-')

plt.scatter(best_k, wcss[best_k-1], color='red', s=120, zorder=5, label=f"Mejor k = {best_k}")
plt.axvline(best_k, color='red', linestyle='--', alpha=0.6)

plt.xlabel('Número de Clústeres (k)')
plt.ylabel('WCSS')
plt.title('Método Elbow')
plt.legend()
plt.grid(True)
plt.show()
