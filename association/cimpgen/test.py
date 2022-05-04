import pandas as pd

df = pd.read_csv('../mutfeats.csv', index_col = 0)
df = df[{"SEC24A_GRCh38_5:134727153-134727153_3'UTR_DEL_T-T--", "RPL22_GRCh38_1:6197725-6197725_Frame-Shift-Del_DEL_T-T--", "class"}]

df = df.loc[(df==1).all(axis=1)]

print(len(df))