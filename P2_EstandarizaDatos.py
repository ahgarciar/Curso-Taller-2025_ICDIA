import pandas as pd
from joblib import dump

instancia = pd.read_csv("PokemonMod.csv")
#print(instancia)

X = instancia.iloc[:,:-1]
Y = pd.DataFrame(instancia.iloc[:,-1])
print(X)
Xarray = X.to_numpy()

from sklearn.preprocessing import StandardScaler as scaler
#from sklearn.preprocessing import MinMaxScaler as scaler #feature_range

x_scaler = scaler().fit(Xarray) #feature_range=(0, 10)
X_array_s = x_scaler.transform(Xarray)

Xstd = pd.DataFrame(data=X_array_s, columns=[X.columns])
Xstd["class"] = Y

Xstd.to_csv("PokemonModSTD.csv", index=None)

print()

dump(x_scaler, "x_scaler.joblib")