import pandas as pd 
from pandas import DataFrame

df = pd.read_csv('data/raw_data.csv', index_col = 0)

cimpp = len(df[df['Class'] == 1])
cimpn = len(df[df['Class'] == 0]) + len(df[df['Class'] == 2])

totalarray = []
for mut in df.columns:
    if mut != 'Class':
        cimppval = df[df["Class"] == 1][mut].sum()
        cimpnval = df[df["Class"] == 0][mut].sum() + df[df["Class"] == 2][mut].sum()
        
        avgcimpp = cimppval/cimpp
        avgcimpn = cimpnval/cimpn
        fold = abs((avgcimpn - avgcimpp) / avgcimpp)
        totalarray.append([mut.split("|")[0], avgcimpp, avgcimpn, cimppval/cimpp - avgcimpn, fold])

df = DataFrame(totalarray, columns=['Gene', 'Cimp+ Average', 'Cimp- Average', 'Average Difference', 'Fold Change'])
df.to_csv("table.csv", index = False)