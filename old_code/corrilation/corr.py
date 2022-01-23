import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.random as nprn
import seaborn as sb



def scatterplot(scattername,scattervalue):
    sortedscattername = []
    sortedscattervalue = []
    counter = 0
    while len(sortedscattervalue) != len(scattervalue):
        index = -1
        maximum = -1
        name = ''
        for x in range(len(scattervalue)):
            if scattervalue[x] > maximum and scattervalue[x] != -2:
                maximum = scattervalue[x]
                name = scattername[x]
                index = x

        sortedscattervalue.append(maximum)
        sortedscattername.append(name)
        scattervalue[index] = -2

        counter += 1
        if counter % 1000 == 0 % 1000:
            print(counter,"/",len(scattervalue))

    print("\nAt Scatter")
    plt.xticks(rotation='vertical')
    plt.scatter(sortedscattername, sortedscattervalue)
    plt.ylabel('Corrilation Values',size = 20)
    plt.savefig("outdir/scatter.png")


def violinplot(scattervalue):
    plt.figure(figsize=(10, 10))
    plt.xticks(rotation='vertical')
    plt.violinplot(scattervalue, vert = False, showmeans = True)
    plt.title('Violin Amount by Correlation Value',size = 20)
    plt.xlabel('Correlation Value',size = 20)
    plt.ylabel('Probability Density',size = 20)
    plt.savefig("outdir/violinplot.png")
    plt.clf()

def Boxplot(scattervalue):
    plt.figure(figsize=(10, 10))
    plt.xticks(rotation='vertical')
    plt.boxplot(scattervalue, vert = False, showmeans = True)
    plt.title('Boxplot Amount by Correlation Value',size = 20)
    plt.xlabel('Correlation Value',size = 20)
    plt.ylabel('Mutations Average',size = 20)
    plt.savefig("outdir/boxplot.png")
    plt.clf()

def boxes(corr, names, mutarraytp, mutarrayfp):

    shortennames = []
    for x in names:
        shortennames.append(x.split('_')[0])

    tparray = []
    for x in range(len(names)):
            tparray.append([shortennames[x], mutarraytp[x]*10])
    outdir =  "outdir/sizekey.csv"
    dataframe = pd.DataFrame(tparray)
    dataframe = dataframe.rename(columns={0: 'source',1: 'TP'})
    dataframe.to_csv(outdir,index = False, header=True)

    tparray = []
    for x in range(len(names)):
            if (mutarraytp[x] - mutarrayfp[x]) < 1:
                tparray.append([shortennames[x], 1])
            else:
                tparray.append([shortennames[x], (mutarraytp[x] - mutarrayfp[x]) *10])
    outdir =  "outdir/sizekey_posneg.csv"
    dataframe = pd.DataFrame(tparray)
    dataframe = dataframe.rename(columns={0: 'source',1: 'TP - FP'})
    dataframe.to_csv(outdir,index = False, header=True)

    start = []
    end = [53,107,143,len(names)]
    cytoval = [0.4,0.45,0.4,0.5]

    for x in range(len(start)):
        box = corr.iloc[start[x]:end[x], start[x]:end[x]]
        box[box <= cytoval[x]] = 0
        string = 'outdir/box' + str(x + 1) + '.csv'
        box.to_csv (string, index = True, header=True)

        ax = sb.heatmap(box, xticklabels = shortennames[start[x]:end[x]], yticklabels = shortennames[start[x]:end[x]])
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        plt.tight_layout()
        string = 'outdir/box' + str(x + 1) + '.png'
        plt.savefig(string)
        plt.clf()

        string = 'cyto' + str(x + 1)
        cytoscape(box, shortennames[start[x]:end[x]],mutarraytp[start[x]:end[x]], string, cytoval[x])

def cytoscape(box, names, mutarraytp, name, cutoff):
    newbox = []
    for x in range(len(box)):
        for y in range(len(box)):
            if(x < y and box.iloc[x][y] > cutoff):
                newbox.append([names[x], box.iloc[x][y] * 10, names[y]])
    outdir =  "outdir/" +name +".csv"
    dataframe = pd.DataFrame(newbox)
    dataframe = dataframe.rename(columns={0: 'source',1: 'weight', 2: 'target'})
    dataframe.to_csv(outdir,index = False, header=True)



def highlow(data,names,num):

    needed_value = 20

    shortennames = []
    for x in names:
        shortennames.append(x.split('_')[0])

    if(num == 0):
        highest = [-1] * needed_value
        result = [-1] * needed_value

        for y in range(0, len(data)):
            for x in range(0, len(data[y])):
                num = data[y][x]
                min_index = highest.index(min(highest))
                min_value = highest[min_index]
                if min_value < num:
                    highest[min_index] = num
                    result[min_index] = (x, y)

        totalarray = []
        for x in result:
	        totalarray.append([names[x[0]], names[x[1]], str(data[x[1],x[0]])])
        pd.DataFrame(totalarray).to_csv("outdir/high.csv",index = False, header=False)

        totalarray = []
        for x in result:
	        totalarray.append([shortennames[x[0]], shortennames[x[1]], str(data[x[1],x[0]]* 10)])
        pd.DataFrame(totalarray).to_csv("outdir/cyto_high.csv",index = False, header=False)

    if(num == 1):
        lowest = [1] * needed_value
        lresult = [1] * needed_value

        for y in range(0, len(data)):
            for x in range(0, len(data[y])):
                num = data[y][x]
                max_index = lowest.index(max(lowest))
                max_value = lowest[max_index]
                if max_value > num:
                    lowest[max_index] = num
                    lresult[max_index] = (x, y)

        totalarray = []
        for x in lresult:
	        totalarray.append([names[x[0]], names[x[1]], str(data[x[1],x[0]])])
        pd.DataFrame(totalarray).to_csv("outdir/low.csv",index = False, header=False)

        totalarray = []
        for x in lresult:
	        totalarray.append([shortennames[x[0]], shortennames[x[1]], str(data[x[1],x[0]]* -10)])
        pd.DataFrame(totalarray).to_csv("outdir/cyto_low.csv",index = False, header=False)


##########################################################
# a path was given to the 0/1 matrix
featmat = pd.read_csv("mutfeats-filteredbyoccurrence.csv", index_col=0)


print("\nAt Correlation")
mutarraytp = []
mutarrayfp = []
for mut in featmat.columns:
        if mut != 'class':
            TP = len(featmat.loc[(featmat["class"] == 1.0) & (featmat[mut] == 1.0)])
            FP = len(featmat.loc[(featmat["class"] != 1.0) & (featmat[mut] == 1.0)])
            mutarraytp.append(TP)
            mutarrayfp.append(FP)

featmat = featmat[featmat["class"] == 1.0]  # drop negative rows
featmat = featmat.drop(columns=['class'])  # drop 'class' column
corr = featmat.corr(method='pearson')
sb.heatmap(corr, yticklabels = False, xticklabels = False)
plt.savefig('outdir/corrmatrix.png')
plt.clf()


names = corr.columns.values.tolist()
data = corr.iloc[0:, 0:].values


print("\nAt Boxes")
boxes(corr, names, mutarraytp, mutarrayfp)

print("\nAt Data Build")
scattername = []
scattervalue = []
for x in range(len(data)):
    for y in range(len(data)):
        if y <= x:
            data[x, y] = 0

for x in range(len(data)):
    for y in range(len(data)):
        if data[x, y] != 0:
            scattername.append(names[x] + '|-----|' + names[x])
            scattervalue.append(data[x, y])

print("\nAt violinplot")
violinplot(scattervalue)

print("\nAt Boxplot")
Boxplot(scattervalue)

print("\nAt High")
highlow(data,names,0)

print("\nAt Low")
highlow(data,names,1)

print("\nAt Scatterplot Sort")
scatterplot(scattername,scattervalue)
