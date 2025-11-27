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

# Metodo del codo con distancia a la recta
k_arr = np.array(list(k_values))
wcss_arr = np.array(wcss)

# Punto inicial y final de la recta
x1, y1 = k_arr[0], wcss_arr[0]
x2, y2 = k_arr[-1], wcss_arr[-1]

# Cálculo de la distancia de cada punto a la recta
numerador = np.abs((y2 - y1) * k_arr - (x2 - x1) * wcss_arr + x2 * y1 - y2 * x1)
denominador = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
distancias = numerador / denominador

best_k = k_arr[np.argmax(distancias)]
print(f"\nMejor valor de k: {best_k}")

# Valores auxiliares para graficar la recta
recta_y = y1 + (y2 - y1) * (k_arr - x1) / (x2 - x1)

plt.figure(figsize=(8, 5))

# Curva de WCSS
plt.plot(k_arr, wcss_arr, marker='o', linestyle='-', label="WCSS")
# Línea recta
plt.plot(k_arr, recta_y, color='gray', linestyle='--', label="Recta Elbow")
# marcar el mejor k
plt.scatter(best_k, wcss_arr[best_k-1], color='red', s=120,
            zorder=5, label=f"Mejor k = {best_k}")
plt.axvline(best_k, color='red', linestyle='--', alpha=0.6)

plt.xlabel('Número de Clústeres (k)')
plt.ylabel('WCSS')
plt.title('Método Elbow')
plt.legend()
plt.grid(True)
plt.show()
