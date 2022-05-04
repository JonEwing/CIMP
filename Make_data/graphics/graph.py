import pandas as pd
import seaborn as sns; sns.set_theme(color_codes=True)
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv("../CIMP_reduced.csv")
df = pd.read_csv("../CIMP_reduced.csv")
samplist = list(set(df["icgc_sample_id"].to_list()))
totalmeans = []
donorlist = []
heatmap = []
namemap= []
for x in samplist:
    samp = df[df["icgc_sample_id"] == x]
    donor = list(set(samp["icgc_donor_id"].to_list()))
    sns.set_theme(style="whitegrid")

    ax = sns.violinplot(y=samp["methylation_value"]).set_title(x)
    plt.savefig('output/' + x + '.png')
    plt.clf()

    donorlist.append(donor)
    totalmeans.append(samp['methylation_value'].mean())
    heatmap.append(samp["methylation_value"].to_list())
    namemap.append(samp["probe_id"].to_list())

# nf = pd.DataFrame(heatmap, index=[samplist], columns=[namemap[0]])
# nf = nf.T
# nf["var"]= nf.var(axis=1).to_list()
# print(nf["var"])
# #nf.sort_values(["var"], ascending=False)
# #nf = nf.head(500)
# nf.to_csv("probes.csv")


df = pd.read_csv("probes.csv", index_col = 0)
df = df.drop(["var"], axis = 1)
samplist= df.columns.to_list()
problist = df.index.to_list()
df = df.T
# nf = pd.DataFrame(donorlist, columns=['Donor_ID'])
# nf["Sample_ID"] = samplist
# nf["Sample_Mean"] = totalmeans
# nf.to_csv("methmeans.csv")

#fig, ax = plt.subplots(figsize=(16, 9))
g = sns.clustermap(df, col_cluster=False, yticklabels = samplist, figsize=(16, 9))
plt.xlabel("Probe Number")
plt.ylabel("Sample ID") 
plt.savefig('clustermap.png')
plt.clf()

X = np.array(df)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

nf = pd.DataFrame(donorlist, columns=['Donor_ID'])
nf["Sample_ID"] = samplist
nf["Sample_Mean"] = df.mean(axis=1).to_list()
nf["ClusterNumber"] = kmeans.labels_
nf.to_csv("Kmeans.csv")


df["Sample_Mean"] = df.mean(axis=1).to_list()
df["ClusterNumber"] = kmeans.labels_
df = df.sort_values(["Sample_Mean"], ascending=False)

clustvals = df["ClusterNumber"].to_list()
df = df.drop(["Sample_Mean","ClusterNumber"], axis = 1)
# nf = pd.DataFrame(donorlist, columns=['Donor_ID'])
# nf["Sample_ID"] = samplist
# nf["Sample_Mean"] = totalmeans
# nf["ClusterNumber"] = kmeans.labels_
# nf["Heatmap"] = heatmap

# nf = nf.sort_values(by=['Sample_Mean'], ascending=False)

fig, ax = plt.subplots(figsize=(16,9))
g = sns.heatmap(df, center=0.5)
plt.xlabel("Probe Number")
plt.ylabel("Sample ID") 

counter = 0
for tick_label in g.axes.get_yticklabels():

    if clustvals[counter] == 1:
        tick_label.set_color("Green")
    elif clustvals[counter] == 0:
        tick_label.set_color("Red")
    else:
        tick_label.set_color("Gold")
    counter += 1

plt.savefig('Heatmap.png')
plt.clf()

plot = sns.clustermap(df, col_cluster=False, figsize=(16, 9), yticklabels=True)
plt.xlabel("Probe Number")
plt.ylabel("Sample ID") 

for tick_label in plot.ax_heatmap.axes.get_yticklabels():
    tick_text = tick_label.get_text()
    for x in range(len(df.index)):
        if df.index.to_list()[x] == tick_text:
            if clustvals[x] == 1:
                tick_label.set_color("Green")
            elif clustvals[x] == 0:
                tick_label.set_color("Red")
            else:
                tick_label.set_color("Gold")
plt.savefig('testmap.png')
plt.clf()