import pandas as pd

forrest = pd.read_csv("150_forrest.csv")
forrest = forrest["Mutation"].tolist() 
ffeature = pd.read_csv('Feature_list.txt', header = None)
ffeature = ffeature[0].tolist() 
tp6 = pd.read_csv("filteredtp6.csv")
tp6 = tp6.columns.tolist()

print("Len of foreest:", len(forrest))
print("Len of feature list:", len(ffeature))
print("Len of tp>6:", len(tp6),"\n")

forrestgenes = []
for x in forrest:
    match = False
    x = x.split("_")[0]
    for y in forrestgenes:
        if x == y:
            match = True
    if match == False:
        forrestgenes.append(x)

tp6genes = []
for x in tp6:
    match = False
    x = x.split("_")[0]
    for y in tp6genes:
        if x == y:
            match = True
    if match == False:
        tp6genes.append(x)

tp6genes = tp6genes[1:len(tp6genes)-1]
print("Len of genes in foreest:", len(forrestgenes))
print("Len of genes in feature list:", len(ffeature))
print("Len of genes in tp>6:", len(tp6genes),"\n")

total = set(forrestgenes + ffeature + tp6genes)


print(len(total))
print(len(total) == len(set(total)))

df = pd.DataFrame(total)
df.to_csv("genes.csv", index = False, header = False)