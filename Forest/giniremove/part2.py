import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#"AE", "FP", "PValue", "CHISquared", "TP-2FP", "Forest", "FP0TP", "Union"

runtype = ["Union", "Chi7.68", "Chi15.36", "Pval0.01", "Pval0.05"]

for params in runtype:
    patha = "out/" + params + ".csv"

    df = pd.read_csv(patha)
    names = df["Mutation"].to_list()
    values = df["Gini_AVG"].to_list()

    figure(figsize=(48, 27))
    temp = plt.bar(names, values, align='center')
    temp[51].set_color('r')
    plt.xticks(names, rotation='vertical')
    plt.savefig("out/" + params + ".png")
    plt.clf()

totalname = []
for params in runtype:
    group = []
    patha = "out/"+params+".csv"
    df = pd.read_csv(patha)

    df = df.nlargest(50,'Gini0')
    names = df["Mutation"].to_list()
    totalname += names

totalname = list(set(totalname))
df = pd.DataFrame(list(totalname), columns=["Mutation"])
df.to_csv("top_50_select.csv", index=False)

df = pd.read_csv("results.csv")

df = df.query('Feature in @totalname')

df.to_csv("Results_gini_genes.csv", index=False)