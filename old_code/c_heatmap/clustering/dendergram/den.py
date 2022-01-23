import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster
from scipy.spatial.distance import pdist,squareform


df = pd.read_csv('mutfeats.csv',index_col=0)
df = df.loc[df["class"] == 1.0]
df = df.drop(["class"], axis=1)


distance = squareform(pdist(df, metric='jaccard'))
linkage = hcluster.complete(distance) #doesnt matter what linkage...
plt.figure(figsize=(10, 3))
plt.title('Hierarchical Clustering Dendrogram Samples')
plt.xlabel('Samples index')
plt.ylabel('distance (Jaccard)')
dendrogram(linkage, labels=df.index, leaf_rotation=90)
plt.savefig("samples.png")
plt.clf

temp = pd.DataFrame(distance)
temp.to_csv('samples_dist.csv', index=False, header = False)


shortennames = []
for x in df.T.index:
    shortennames.append(x.split('_')[0])


distance = squareform(pdist(df.T, metric='jaccard'))
linkage = hcluster.complete(distance) #doesnt matter what linkage...
plt.figure(figsize=(10, 3))
plt.title('Hierarchical Clustering Dendrogram Samples')
plt.xlabel('Samples index')
plt.ylabel('distance (Jaccard)')
dendrogram(linkage, labels=shortennames, leaf_rotation=90)
plt.savefig("mutations.png")
plt.clf

temp = pd.DataFrame(distance)
temp.to_csv('mutations_dist.csv', index=False, header = False)