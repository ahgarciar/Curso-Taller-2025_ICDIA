import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("PokemonMod.csv")
    X = df.values

    for pokemon in X:
        print(pokemon)