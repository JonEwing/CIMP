import pandas as pd
import numpy


df = pd.read_csv("simple_somatic_mutation.open.tsv", sep = '\t')
chrom = df["chromosome"].to_list()
start = df["chromosome_start"].to_list()
end = df["chromosome_end"].to_list()
samps = df["icgc_sample_id"].to_list()

mf = pd.read_csv("mutss.txt", sep = '\t')
mf = mf.fillna("")

xval = mf["Name1"].to_list()
yval = mf["Name2"].to_list()

idxDict= dict()
for x in range(len(xval)):
    idxDict[xval[x]] = yval[x]

df['gene_affected'] = df['gene_affected'].map(idxDict)
mutss = df["gene_affected"].to_list()

kf = pd.read_csv("kmeans.csv")
name = kf["Sample_ID"].to_list()
kmeans = kf["ClusterNumber"].to_list()

xaxis = []
for x in range(len(start)):
    xaxis.append(str(chrom[x]) + ":" + str(start[x]) + "-" + str(end[x]))
    #xaxis.append(str(mutss[x]))
xaxis = list(set(xaxis))

totalvector = []
for x in range(len(start)):
    totalvector.append([samps[x], str(chrom[x]) + ":" + str(start[x]) + "-" + str(end[x])])
    #totalvector.append([samps[x], str(mutss[x])])

mutfeat = numpy.zeros((len(list(set(name))), len(xaxis)))

df = pd.DataFrame(mutfeat, columns = xaxis, index = name)

counter = 0
for x in totalvector:
    df.at[x[0], x[1]] = 1
    if counter % 10000 == 0:
        print(str(counter) + " / " + str(len(totalvector)))
    counter  += 1

#df = df[df.columns[df.sum()>2]]
try:
    df = df.drop([""], axis = 1)
except:
    pass

print("Cols", len(df.columns.to_list()))

df["class"] = kmeans

df.to_csv('mutfeats_new.csv')