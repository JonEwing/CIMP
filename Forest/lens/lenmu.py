import pandas as pd

dataset = pd.read_csv('../data/mutfeats no cimp class labels.csv',index_col=0)

occur4 = []

for sample in dataset.index:
    TP = 0
    for mut in dataset.columns:
        if dataset.loc[sample,mut] == 1:
            TP += 1
    occur4.append([sample, TP])

tf = pd.DataFrame(occur4, columns = ["Name", "Value"])
tf.to_csv('lens.csv', index=False)

uniquenums = []
for x in range(len(occur4)):
    match = False
    for y in range(len(uniquenums)):
        if uniquenums[y][0] == occur4[x][1]:
            match = True
    if match == False:
        uniquenums.append([occur4[x][1]])

newtotals = []
for x in range(len(uniquenums)):
    newtotals.append([uniquenums[x][0], 0, 0])
    for y in range(len(occur4)):
        if newtotals[x][0] == occur4[y][1] and occur4[y][0][0] == 'C':
            newtotals[x][1] += 1
        if newtotals[x][0] == occur4[y][1] and occur4[y][0][0] == 'N':
            newtotals[x][2] += 1

tf = pd.DataFrame(newtotals, columns = ["Value", "Cancer", "Non-Cancer"])
tf.to_csv('counterlens.csv', index=False)