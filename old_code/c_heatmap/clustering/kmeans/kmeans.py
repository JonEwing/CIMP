import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sb

df = pd.read_csv('mutfeats.csv',index_col=0)
#df = df.loc[df["class"] == 1.0]
df = df.drop(["class"], axis=1)

#df.to_csv("cimppos.csv")

colnames = df.columns
clustergroups = []
non0names = df.index.values
Histnums = []

data = df.values


sb.clustermap(df.T, xticklabels = data, figsize=(30,30), cmap = "BuPu")
plt.savefig('tmp.png',dpi=150)


trials = 0

while trials < 5:
    print("\n\nTest ", trials + 1)

    np1 = random.randint(0, len(data)-1)
    np2 = random.randint(0, len(data)-1)
    np3 = random.randint(0, len(data)-1)
        
    while np1 == np2 or np2 == np3:
        np2 = random.randint(0, len(data)-1)
        np3 = random.randint(0, len(data)-1)

    oldnp1 = 0
    oldnp2 = 0 
    oldnp3 = 0

    i = 0
    while i < 15:
        indext = 0
        cluster1 = []
        cluster2 = []
        cluster3 = []

        j = 0
        for x in data:
            
            test1 = np.argwhere(x == 1)

            try:
                test2 = np.argwhere(data[np1] == 1)
                intersect = np.intersect1d(test1,test2)
                Union = np.union1d(test1,test2)
                dist1 = (1 - (len(intersect) / len(Union)))
            except:
                dist1 = 0

            try:
                test2 = np.argwhere(data[np2] == 1)
                intersect = np.intersect1d(test1,test2)
                Union = np.union1d(test1,test2)
                dist2 = (1 - (len(intersect) / len(Union)))
            except:
                dist2 = 0

            try:
                test2 = np.argwhere(data[np3] == 1)
                intersect = np.intersect1d(test1,test2)
                Union = np.union1d(test1,test2)
                dist3 = (1 - (len(intersect) / len(Union)))
            except:
                dist3 = 0


            if dist1 <= dist2 and dist1 <= dist3:
                cluster1.append([x,j])
            elif dist2 < dist1 and dist2 < dist3:
                cluster2.append([x,j])
            elif dist3 < dist1 and dist3 < dist2:
                cluster3.append([x,j])
            j += 1
            
        clust1dist = []
        for x in cluster1:
            dist = 0
            for y in cluster1:
                test1 = np.argwhere(x[0] == 1)
                test2 = np.argwhere(y[0] == 1)
                try:
                    intersect = np.intersect1d(test1,test2)
                    Union = np.union1d(test1,test2)
                    dist = (1 - (len(intersect) / len(Union)))
                except:
                    dist = 0
            clust1dist.append(dist)

        indext += np.sum(clust1dist)/ len(clust1dist)
        index = np.argmin(clust1dist)
        np1 = cluster1[index][1]

        clust2dist = []
        for x in cluster2:
            dist = 0
            for y in cluster2:
                test1 = np.argwhere(x[0] == 1)
                test2 = np.argwhere(y[0] == 1)

                try:
                    intersect = np.intersect1d(test1,test2)
                    Union = np.union1d(test1,test2)
                    dist = (1 - (len(intersect) / len(Union)))
                except:
                    dist = 0
            clust2dist.append(dist)

        indext += np.sum(clust2dist)/ len(clust2dist)
        index = np.argmin(clust2dist)
        np2 = cluster2[index][1]

        clust3dist = []
        for x in cluster3:
            dist = 0
            for y in cluster3:
                test1 = np.argwhere(x[0] == 1)
                test2 = np.argwhere(y[0] == 1)

                try:
                    intersect = np.intersect1d(test1,test2)
                    Union = np.union1d(test1,test2)
                    dist = (1 - (len(intersect) / len(Union)))
                except:
                    dist = 0
            clust3dist.append(dist)

        indext += np.sum(clust3dist)/ len(clust3dist)
        index = np.argmin(clust3dist)
        np3 = cluster3[index][1]

        i += 1

        if oldnp1 == np1  and oldnp2 == np2 and oldnp3 == np3:
            clustergroups.append([indext, cluster1, cluster2, cluster3])
            break

        oldnp1 = np1
        oldnp2 = np2
        oldnp3 = np3

    trials += 1
    print(indext)

min = 100
for x in clustergroups:
    if x[0] < min:
        min = x[0]
        bestnp = x



cluster1 = bestnp[1]
cluster2 = bestnp[2]
cluster3 = bestnp[3]

num1 = []
num2 = []
num3 = []
names1 = []
names2 = []
names3 = []


for x in cluster1:
    names1.append(non0names[x[1]])
    num1.append(x[0])
    
for x in cluster2:
    names2.append(non0names[x[1]])
    num2.append(x[0])

for x in cluster3:
    names3.append(non0names[x[1]])
    num3.append(x[0])


print("length of cluster 1: ",len(cluster1))
print("length of cluster 2: ",len(cluster2))
print("length of cluster 3: ",len(cluster3))

df1 = pd.DataFrame(num1)
df1.columns = colnames
plt.figure(figsize=(30,30))
plt.title("Cluster 1", size = 50)
heat_map = sb.heatmap(df1, square = True, yticklabels = names1, xticklabels = Histnums, cbar_kws=({"shrink": .80}), cmap = "BuPu")
plt.savefig("./files/clust1.png")
plt.clf()

df2 = pd.DataFrame(num2)
df2.columns = colnames
plt.figure(figsize=(30,30))
plt.title("Cluster 2", size = 50)
heat_map = sb.heatmap(df2, square = True, yticklabels = names2, xticklabels = Histnums, cbar_kws=({"shrink": .80}), cmap = "BuPu")
plt.savefig("./files/clust2.png")
plt.clf()

df3 = pd.DataFrame(num3)
df3.columns = colnames
plt.figure(figsize=(30,30))
plt.title("Cluster 3", size = 50)
heat_map = sb.heatmap(df3, square = True, yticklabels = names3, xticklabels = Histnums, cbar_kws=({"shrink": .80}), cmap = "BuPu")
plt.savefig("./files/clust3.png")
plt.clf()



total = []
for x in colnames:
    a = df1[x].sum()
    b = df2[x].sum()
    c = df3[x].sum()
    total.append([x,a,b,c])

total = pd.DataFrame(data=total, columns= ["Names", "Cluster1", "Cluster2", "Cluster3"])
total.to_csv("total.csv", index = False,)


df = pd.read_csv('mutfeats.csv')
df = df.drop(['class'], axis=1)
df['New'] = [','.join([str(df.columns[x]) for x,y in enumerate(list(i[-1])) if y==1]) for i in df.iterrows()]
liststuff = df['New'].to_list()
listnames = df['Unnamed: 0'].to_list()

total = []
for x in range(len(listnames)):
    stuff = liststuff[x].split(',')
    stuff.insert(0,listnames[x])
    total.append(stuff)

left = []
middle = []
right = []

for x in total:
    for y in names1:
        if x[0] == y:
            left.append(x)
    for y in names2:
        if x[0] == y:
            middle.append(x)
    for y in names3:
        if x[0] == y:
            right.append(x)

df = DataFrame(left)
df.to_csv("./files/Clust1.csv", index = False, header = False)

df = DataFrame(middle)
df.to_csv("./files/Clust2.csv", index = False, header = False)

df = DataFrame(right)
df.to_csv("./files/Clust3.csv", index = False, header = False)