import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataMatrix = pd.read_csv('mutfeats.csv', index_col=0)

dataMatrix = dataMatrix[dataMatrix["class"] == 1]
npdata = dataMatrix.to_numpy()
kmeans = KMeans(n_clusters=4, n_init = 1000).fit(npdata)
clust = list(kmeans.fit_predict(npdata))

dataMatrix["clust"] = clust

colorstore = {}
for index, row in dataMatrix.iterrows():
    if(row['clust'] == 0.0):
        colorstore[index] = "Blue"
    elif(row['clust'] == 1.0):
        colorstore[index] = "Red"
    elif(row['clust'] == 2.0):
        colorstore[index] = "Green"
    else:
        colorstore[index] = "black"

print(clust)

dataMatrix = dataMatrix.drop('clust',1)
df = pd.DataFrame(npdata, columns = dataMatrix.columns.to_list(), index = dataMatrix.index.to_list())


g = sns.clustermap(df.transpose(), figsize=(40,25), yticklabels=True, cmap='viridis')
for tick_label in g.ax_heatmap.axes.get_xticklabels():
    ttxt = tick_label.get_text()
    tick_label.set_color(colorstore[ttxt])
g.dendrogram_col.linkage # linkage matrix for columns
g.dendrogram_row.linkage
plt.savefig("Cimppos.png")