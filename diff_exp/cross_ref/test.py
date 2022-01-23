import pandas as pd


keeps = ["row", "log2FoldChange", "pvalue", "padj"]
keepsother = ["log2FoldChange", "pvalue", "padj"]
df = pd.read_csv("uterine.csv")

dfgas = pd.read_csv("gastric.csv")
data = dfgas["row"].values
df = df[df["row"].isin(data)]

dfcol = pd.read_csv("colrec.csv")
data = dfcol["row"].values
df = df[df["row"].isin(data)]

data = df["row"].values
dfgas = dfgas[dfgas["row"].isin(data)]
dfcol = dfcol[dfcol["row"].isin(data)]

df = df[keeps]
dfgas = dfgas[keeps]
dfcol = dfcol[keeps]
df.to_csv("out/combined_chrommod.csv", index = False)
dfcol.to_csv("out/col.csv", index = False)
dfgas.to_csv("out/gas.csv", index = False)