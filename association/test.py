import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, fpgrowth, apriori

df = pd.read_csv('results.csv', sep=',', index_col = 0)

df['CHI Squared'] = df['CHI Squared'].astype(float)
df['p value'] = df['p value'].astype(float)

list1 = set(df[df['CHI Squared'] > 3].index.to_list())

df = pd.read_csv('results.csv', sep=',', index_col = 0)
list2 = set(df[df['p value'] < 0.005].index.to_list())

df = pd.read_csv('results.csv', sep=',', index_col = 0)
tf = df[df['FP'] == 0]
list3 = set(tf[tf['TP'] > 3].index.to_list())

lista = list1.intersection(list2)
listb = list3.intersection(list2)
muts = lista.union(listb)

muts = list1

print(len(muts))
df = pd.read_csv('mutfeats.csv', sep=',')
df = df[muts] 

num = 4/ len(df.index.to_list())

#################################################################################################################

frequent_itemsets_ap = apriori(df, min_support=num, use_colnames=True)
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
