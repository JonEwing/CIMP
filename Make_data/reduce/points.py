import pandas as pd

cf = pd.read_csv("CIMP.txt", sep='\t')

mf = pd.read_csv("meth.tsv", sep='\t')

df = mf[mf['probe_id'].isin(cf["Data1"].to_list())]

df.to_csv("test.csv")

print(mf)
print(df)