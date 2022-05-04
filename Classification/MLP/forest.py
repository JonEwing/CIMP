import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("../input/results_new.csv")
collist = df["Feature"].tolist()

df = pd.read_csv("../input/results.csv")
utelist = df["Feature"].tolist()

intlist = list(set(collist) & set(utelist))

shutil.rmtree("data")
os.mkdir("data")

df = pd.read_csv("../input/results.csv", index_col=0)
df = df.loc[intlist]

intlist.append("class")

colrecdata = pd.read_csv('../input/mutfeats_new.csv', index_col=0)
colrecdata = colrecdata[colrecdata["class"] != 2]
#colrecdata['class'] = colrecdata['class'].replace([2],-1)
colrecdata = colrecdata[intlist]

utedata = pd.read_csv('../input/mutfeats.csv', index_col=0)
utedata = utedata[utedata["class"] != 2]
#utedata['class'] = utedata['class'].replace([2],-1)
utedata = utedata[intlist]

shutil.rmtree("data")
os.mkdir("data")

df['chi'] = (((df["TP"] - df["TP"].mean()) ** 2) / df["TP"].mean()) + (((df["FP"] - df["FP"].mean()) ** 2) / df["FP"].mean()) + \
    (((df["TN"] - df["TN"].mean()) ** 2) / df["TN"].mean()) + \
    (((df["FN"] - df["FN"].mean()) ** 2) / df["FN"].mean())
df["tp-fp"] = df["TP"] - 2*df["FP"]
fp0 = df[df['FP'] == 0]

#"Pval<0.05", "Pval<0.01", "Pval<0.005", "Chi>3.84", "Chi>7.68", "Chi>15.36", "TP3_FP0", "TP4_FP0", "TP5_FP0"
params = ["Pval<0.05", "Pval<0.01", "Pval<0.005", "Chi>3.84", "Chi>7.68", "Chi>15.36", "TP-2FP>2", "TP-2FP>3", "TP-2FP>4", "TP3_FP0", "TP4_FP0", "FP=0", "All"]


uninonlist = []
runtype = []
for a in params:
    runtype.append([a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

total = []
totalmean = []
totalruns = 0
crossfold = 5

tf = pd.DataFrame()
remover = 0.0
while tf.empty:
    total = []
    totalmean = []
    for parm in runtype:
        valuelist = []
        fprlist = []
        tprlist = []
        group = []
        group1 = []
        patha = "data/"+str(parm[0])+"/"
        try:
            os.mkdir(patha)
        except:
            shutil.rmtree(patha)
            os.mkdir(patha)
        print(parm[0], '\n')
        for cross in range(crossfold):
            path = patha + "Run_"+str(cross)+"/"
            try:
                os.mkdir(path)
            except:
                shutil.rmtree(path)
                os.mkdir(path)

            datasetc = utedata

            if parm[0] == "FP=0":
                sep = df[df["FP"] == 0]
                muts = sep.index.to_list()
            if parm[0] == "FP<=1":
                sep = df[df["FP"] <= 1]
                muts = sep.index.to_list()
            if parm[0] == "FP<=2":
                sep = df[df["FP"] <= 2]
                muts = sep.index.to_list()

            if parm[0] == "Pval<0.05":
                sep = df[df["p value"] < 0.05]
                muts = sep.index.to_list()
            if parm[0] == "Pval<0.01":
                sep = df[df["p value"] < 0.01]
                muts = sep.index.to_list()
            if parm[0] == "Pval<0.005":
                sep = df[df["p value"] < 0.005]
                muts = sep.index.to_list()

            if parm[0] == "Chi>3.84":
                sep = df[df["chi"] > 3.84]
                muts = sep.index.to_list()
            if parm[0] == "Chi>7.68":
                sep = df[df["chi"] > 7.68]
                muts = sep.index.to_list()
            if parm[0] == "Chi>15.36":
                sep = df[df["chi"] > 15.36]
                muts = sep.index.to_list()

            if parm[0] == "TP-2FP>2":
                sep = df[df["tp-fp"] > 2]
                muts = sep.index.to_list()
            if parm[0] == "TP-2FP>3":
                sep = df[df["tp-fp"] > 3]
                muts = sep.index.to_list()
            if parm[0] == "TP-2FP>4":
                sep = df[df["tp-fp"] > 4]
                muts = sep.index.to_list()

            if parm[0] == "TP3_FP0":
                sep = fp0[fp0["TP"] > 3]
                muts = sep.index.to_list()
            if parm[0] == "TP4_FP0":
                sep = fp0[fp0["TP"] > 4]
                muts = sep.index.to_list()
            if parm[0] == "TP5_FP0":
                sep = fp0[fp0["TP"] > 5]
                muts = sep.index.to_list()

            if parm[0] == "All":
                muts = df.index.to_list()

            X = datasetc[muts].to_numpy()

            uninonlist += muts

            ############################################################################################

            y = datasetc['class'].to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1/crossfold), train_size=1 - (1/crossfold))
            
            space = dict()
            space["solver"] = ["adam","sgd"]
            space["learning_rate_init"] = [0.0001]
            space["max_iter"] = [500]
            # space["hidden_layer_sizes"] = [(500, 400, 300, 200, 100), (400, 400, 400, 400, 400), (300, 300, 300, 300, 300), (200, 200, 200, 200, 200)]
            # space["activation"] = ["logistic","tanh","relu"]
            # space["alpha"] = [0.0001, 0.001, 0.005]
            # space["early_stopping"] = [True, False]
   
            #Uncomment both lines if you want to use two datasets
            X_test = colrecdata[muts].to_numpy()
            y_test = colrecdata['class'].to_numpy()

            clf = MLPClassifier()           
            result = GridSearchCV(clf, space)
            result.fit(X_train, y_train)
            best_model = result.best_estimator_
            y_pred = best_model.predict(X_test)
            probl = best_model.predict_proba(X_test)
            probr = best_model.predict_proba(X_test)

            predicted = []
            names = colrecdata.index.to_list()
            for x in range(len(y_test)):
                correct = 0
                if y_test[x] == y_pred[x]:
                    correct = 1
                predicted.append([colrecdata.index.to_list()[x], y_test[x],y_pred[x], probl[x][0], probr[x][1], correct])

            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for x in predicted:
                if x[1] == 1 and x[2] == 1:
                    TP += 1
                if x[1] != 1 and x[2] == 1:
                    FP += 1
                if x[1] != 1 and x[2] != 1:
                    TN += 1
                if x[1] == 1 and x[2] != 1:
                    FN += 1

            parm[1] = cross
            parm[2] = TP
            parm[3] = FP
            parm[4] = FN
            parm[5] = TN
            try:
                parm[6] = (TP + TN) / (TP + FP + TN + FN)
            except:
                parm[6] = -1
            try:
                parm[7] = (TP) / (TP + FN)
            except:
                parm[7] = -1
            try:
                parm[8] = (TN) / (TN + FP)
            except:
                parm[8] = -1
            try:
                parm[9] = (TP) / (TP + FP)
            except:
                parm[9] = -1
            try:
                parm[10] = (FN) / (FN + TP)
            except:
                parm[10] = -1
            try:
                parm[11] = (FP) / (FP + TP)
            except:
                parm[11] = -1
            try:
                parm[12] = (FN) / (FN + TN)
            except:
                parm[12] = -1

            hold = [parm[0], parm[1], parm[2], parm[3], parm[4], parm[5], parm[6], parm[7], parm[8], parm[9], parm[10], parm[11], parm[12], parm[13], parm[14], len(muts)]

            tf = pd.DataFrame(predicted, columns=["Name", "Actual", "Predicted", "Probaility left", "Probaility Right", "Correctly Classified"])
            tf.to_csv(path+'predicted.csv', index=False)

            print(str(cross + 1), "/", crossfold)
            group.append(hold)
            total.append(hold)

        tf = pd.DataFrame(group, columns=["Seperator Type", "Three-Fold Run", "TP", "FP", "FN", "TN", "Accuracy",
                                        "Sensitivity", "Specificity", "Precision", "Miss Rate", "False discovery rate", "False omission rate", "ROC_AUC", "PR Logistic", "Included Muts"])
        tf.to_csv(patha + 'group_stats.csv', index=False)
        tf = tf.drop("Seperator Type", axis = 1)
        tf = tf.drop("Three-Fold Run", axis = 1)
        tfmean = tf.mean().values.tolist()
        totalmean.append([parm[0]] + tfmean)


        jf = pd.DataFrame(muts, columns=["Mutations"])
        jf.to_csv(patha + 'mutations.csv', index=False)
        

        totalruns += 1
        print("\n")
        print(str(totalruns), "/", len(runtype))

        print("############################################\n")

    tf = pd.DataFrame(total, columns=["Seperator Type", "Three-Fold Run", "TP", "FP", "FN", "TN", "Accuracy",
                                    "Sensitivity", "Specificity", "Precision", "Miss Rate", "False discovery rate", "False omission rate", "ROC_AUC", "PR Logistic", "Included Muts"])
    tf.to_csv('total_stat.csv', index=False)

    tf = pd.DataFrame(totalmean, columns=["Seperator Type", "TP", "FP", "FN", "TN", "Accuracy", "Sensitivity",
                                        "Specificity", "Precision", "Miss Rate", "False discovery rate", "False omission rate", "ROC_AUC", "PR Logistic",  "Included Muts"])
    tf.to_csv('total_mean.csv', index=False)

    threshold = 0.1
    tf = tf[tf["Accuracy"] >= threshold - remover]
    
    remover += 0.02
    totalruns = 0
    print(threshold - remover, "\n\n")