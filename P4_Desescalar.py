import pandas as pd
from joblib import load

scaler = load("x_scaler.joblib")

instancia = pd.read_csv("Pokemon-con-clusters.csv")
#print(instancia)

X = instancia.iloc[:,:-2]
r = len(instancia.columns)
Y = pd.DataFrame(instancia.iloc[:,r-2:r])
#print(X)
Xarray = X.to_numpy()

# Desescalar
X_original = scaler.inverse_transform(Xarray)
print(X_original)

instancia = pd.DataFrame(data=X_original, columns=[X.columns])

instancia[["class", "cluster"]] = Y[["class", "cluster"]]

instancia.to_csv("Pokemon-con-clusters-desescalado.csv",index=False)
