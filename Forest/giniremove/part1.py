import pandas as pd
import os
import shutil

try:
    os.mkdir("out")
except:
    shutil.rmtree("out")
    os.mkdir("out")
#"AE", "FP", "PValue", "CHISquared", "TP-2FP", "Forest", "FP0TP", "Union"
params = ["Pval0.05", "Pval0.01", "Pval0.005", "Chi3.84", "Chi7.68", "Chi15.36", "TP3_FP0"]


runtype = []
for a in params:
    runtype.append([a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

crossfold = 11
for params in runtype:
    group = []
    patha = "../multipleparams/data/"+str(params[0])+"/"
    for cross in range(crossfold):
        path = patha +"Run_"+str(cross)+"/"+"feature_importance.csv"
        df = pd.read_csv(path)
        df.sort_values(by=['Mutation'], inplace=True, ascending = False)

        if cross == 0:
            tf = df
            ginisum = df["Value"].to_list()
            name0 = df["Mutation"].to_list()
            tf.rename(columns = {'Value' : 'Gini0'}, inplace = True)

        else:
            tmpgini = df["Value"].to_list()
            tmpname = df["Mutation"].to_list()
            tf['Gini'+ str(cross)] = tmpgini

            for x in range(len(name0)):
                for y in range(len(tmpname)):
                    if name0[x] == tmpname[y]:
                        ginisum[x] += tmpgini[y] 
               
    print(str(params[0]))
    print("###############################################################################")

    tf["Gini_AVG"] = ginisum
    tf["Gini_AVG"] = tf["Gini_AVG"] / 10
    tf.sort_values(by=['Gini_AVG'], inplace=True, ascending = False)
    tf.to_csv('out/' + str(params[0]) + ".csv", index=False)


##########################################################################################################################

totalname = []
for params in runtype:
    group = []
    patha = "out/"+str(params[0])+".csv"
    df = pd.read_csv(patha)

    df = df[df['Gini_AVG']  > 1/ len(df['Gini_AVG'].tolist())]
    names = df["Mutation"].to_list()
    totalname += names

totalname = list(set(totalname))
df = pd.DataFrame(list(totalname), columns=["Mutation"])
df.to_csv("sample.csv", index=False)