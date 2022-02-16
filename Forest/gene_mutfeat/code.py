import pandas as pd

df = pd.read_csv("mutfeats_Ovarian_and_uterine_cell_lines.csv", index_col = 0)
classstore = df["class"].to_list()
df.drop(["class"], axis = 1)
mutnames = df.columns.to_list()
genenames = []

for x in mutnames:
    genenames.append(x.split("_")[0])


mylist = genenames
a = set([i for i in mylist if mylist.count(i)>1])
print (a)

genenames = list(set(genenames))

combined = []
for x in genenames:
    col = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for y in mutnames:
        if x == y.split("_")[0]:
            data = df[y].to_list()
            for z in range(len(data)):
                if data[z] == 1:
                    col[z] = 1
    combined.append(col)

df = pd.DataFrame(combined, index = genenames, columns = df.index.to_list())
df = df.T
df["class"] = classstore
df.to_csv("gene_feats.csv")