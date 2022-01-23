import pandas as pd

filename = "../test_data.csv"
df = pd.read_csv(filename,index_col=0)
total = [["Mutation", "TP", "FP", "FN", "TN"]]
for mut in df.columns:
    if mut == 'class':
        continue

    TP = len(df.loc[(df["class"] == 1.0) & (df[mut] == 1.0)])
    FP = len(df.loc[(df["class"] != 1.0) & (df[mut] == 1.0)])
    FN = len(df.loc[(df["class"] == 1.0) & (df[mut] != 1.0)])
    TN = len(df.loc[(df["class"] != 1.0) & (df[mut] != 1.0)])
    total.append([mut,TP,FP,FN,TN])

df = pd.DataFrame(total)
df.to_csv("output.csv", index=False, header=False)
		