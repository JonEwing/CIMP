import pandas as pd

df = pd.read_csv('results.csv', sep=',')

df['chi'] = (((df["TP"] - df["TP"].mean()) ** 2) / df["TP"].mean()) + (((df["FP"] - df["FP"].mean()) ** 2) / df["FP"].mean()) + \
    (((df["TN"] - df["TN"].mean()) ** 2) / df["TN"].mean()) + \
    (((df["FN"] - df["FN"].mean()) ** 2) / df["FN"].mean())

sep = df[df["chi"] > 15]
totalmuts = sep["Feature"].to_list()

#totalmuts = pd.read_csv('mut_list.csv', sep=',')
#totalmuts = totalmuts["Mutations"].to_list()

#df = df.drop("class", axis = 1)

lista = ["Sum", "Cluster", "Subtype", "MSIInfoGain", "MSICartPhi", "MSI", "class"]
df = pd.read_csv('test.csv', sep=',', index_col = 0)
df = df.filter(lista + totalmuts)

df.to_csv("test2.csv")