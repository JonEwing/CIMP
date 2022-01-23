import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import savetxt
import seaborn as sb



df = pd.read_csv('mutfeats.csv')

df = df.drop(['class', 'Unnamed: 0'], axis=1)


size = len(np.argwhere(df.values.tolist()))

supportgraph = []
cofidencegraph = []
liftgraph = []
for x in df.columns:
    tempsupport = []
    tempconfidence = []
    templift = []
    for y in df.columns:
        if x != y:
            col1 = df[x].values.tolist()
            col1 = np.argwhere(col1)
            col2 = df[y].values.tolist()
            col2 = np.argwhere(col2)
            intersect = np.intersect1d(col1,col2)
            tempsupport.append(len(intersect)/size)
            tempconfidence.append(len(intersect)/len(col1))
            templift.append(len(intersect)/(len(col1) * len(col2)))
        else:
            tempsupport.append(0)
            tempconfidence.append(0)
            templift.append(0)
    supportgraph.append(tempsupport)
    cofidencegraph.append(tempconfidence)
    liftgraph.append(templift)
        

names = list(df.columns)
df = pd.DataFrame(supportgraph, index = names, columns= names)
df.to_csv("assosiation/support.csv")

sb.heatmap(supportgraph, yticklabels = False, xticklabels = False)
plt.savefig('assosiation/support.png')
plt.clf()


df = pd.DataFrame(cofidencegraph, index = names, columns= names)
df.to_csv("assosiation/confidence.csv")

sb.heatmap(cofidencegraph, yticklabels = False, xticklabels = False)
plt.savefig('assosiation/confidence.png')
plt.clf()


df = pd.DataFrame(liftgraph, index = names, columns= names)
df.to_csv("assosiation/lift.csv")

sb.heatmap(liftgraph, yticklabels = False, xticklabels = False)
plt.savefig('assosiation/lift.png')
plt.clf()