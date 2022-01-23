import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, fpgrowth, apriori


df = pd.read_csv('results.csv', sep=',')

df['chi'] = (((df["TP"] - df["TP"].mean()) ** 2) / df["TP"].mean()) + (((df["FP"] - df["FP"].mean()) ** 2) / df["FP"].mean()) + \
    (((df["TN"] - df["TN"].mean()) ** 2) / df["TN"].mean()) + \
    (((df["FN"] - df["FN"].mean()) ** 2) / df["FN"].mean())
fp0 = df[df['FP'] == 0]

totalmuts = []

sep = df[df["FP"] == 0]
totalmuts += sep["Feature"].to_list()

sep = df[df["chi"] > 3.84]
totalmuts += sep["Feature"].to_list()

totalmuts = list(set(totalmuts))

pd.DataFrame(totalmuts, columns=["Mutations"]).to_csv('mut_list.csv', index=False)

df = pd.read_csv('mutfeats.csv', sep=',', index_col = 0)

df = df.drop("class", axis = 1)
df = df.filter(totalmuts)

cols = df.columns.to_list()

rows = df.values.tolist()

recored = []
for i in range(len(rows)):
    recored.append([int(df.values[i,j]) for j in range(len(cols))])


df = pd.DataFrame(recored, columns=cols)

#################################################################################################################

frequent_itemsets_ap = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets_ap['itemsets'] = frequent_itemsets_ap['itemsets'].apply(set)
length = []
for x in frequent_itemsets_ap['itemsets'].apply(set):
    length.append(len(x))
frequent_itemsets_ap['Item_len'] = length
pd.DataFrame(frequent_itemsets_ap).to_csv('out/frequent_itemsets_ap.csv', index=False)

#################################################################################################################

rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.8)
rules_ap['antecedents'] = rules_ap['antecedents'].apply(set)
length = []
for x in rules_ap['antecedents'].apply(set):
    length.append(len(x))
rules_ap['antecedents_len'] = length
rules_ap['consequents'] = rules_ap['consequents'].apply(set)
length = []
for x in rules_ap['consequents'].apply(set):
    length.append(len(x))
rules_ap['consequents_len'] = length
rules_ap['Total_len'] = rules_ap['consequents_len'] + rules_ap['antecedents_len']
pd.DataFrame(rules_ap).to_csv('out/rules_ap.csv', index=False)
