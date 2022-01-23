import pandas as pd

df = pd.read_csv("results_C.csv")
collist = df["Feature"].tolist()

df = pd.read_csv("results_J.csv")
utelist = df["Feature"].tolist()

intlist = list(set(collist) & set(utelist))

print(len(intlist))