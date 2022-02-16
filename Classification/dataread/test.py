import pandas as pd


params = ["Pval0.05", "Pval0.01", "Pval0.005", "Chi3.84", "Chi7.68", "Chi15.36", "TP-2FP2", "TP-2FP3", "TP-2FP4", "TP3_FP0", "TP4_FP0", "FP=0", "All"]

path = "../SVM/data/Pval0.05/Run_0/predicted.csv"
df = pd.read_csv(path, index_col=0)
storednames = df.index.to_list()

values = []
values.append([0] * len(storednames))
values.append([0] * len(storednames))

for x in params:
    for y in range(10):
        path = "../SVM/data/"+x+"/Run_"+str(y)+"/predicted.csv"
        df = pd.read_csv(path, index_col=0)
        temp = df["Correctly Classified"].to_list()
        for z in range(len(temp)):
            if temp[z] == 0:
                values[0][z] += 1
            else:
                values[1][z] += 1

tf = pd.DataFrame(storednames, columns=["Sample"])
tf["Incorrect"] = values[0]
tf["Correct"] = values[1]
tf.to_csv('Sample_Correctness.csv', index=False) 