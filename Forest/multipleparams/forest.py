import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

def plot_roc_curve(fpr, tpr, Run, name):
    plt.plot(fpr, tpr, color="orange", label="Fold:" + str(Run))
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_auc/" + str(name) + "_" + str(Run) + ".png")
    plt.clf()
    
def plot_pr_curve(test_y, model_probs, Run, name):
    no_skill = len(test_y[test_y == 1]) / len(test_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    precision, recall, _ = precision_recall_curve(test_y, model_probs)
    plt.plot(recall, precision, marker='.', label='Logistic')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig("pr/" + str(name) + "_" + str(Run) + ".png")
    plt.clf()


shutil.rmtree("data")
shutil.rmtree("roc_auc")
shutil.rmtree("pr")
os.mkdir("data")
os.mkdir("roc_auc")
os.mkdir("roc_auc/roc_median")
os.mkdir("pr")

df = pd.read_csv("input/results.csv")

df['chi'] = (((df["TP"] - df["TP"].mean()) ** 2) / df["TP"].mean()) + (((df["FP"] - df["FP"].mean()) ** 2) / df["FP"].mean()) + \
    (((df["TN"] - df["TN"].mean()) ** 2) / df["TN"].mean()) + \
    (((df["FN"] - df["FN"].mean()) ** 2) / df["FN"].mean())
df["tp-fp"] = df["TP"] - 2*df["FP"]
fp0 = df[df['FP'] == 0]

#"Pval<0.05", "Pval<0.01", "Pval<0.005", "Chi>3.84", "Chi>7.68", "Chi>15.36", "TP3_FP0", "TP4_FP0", "TP5_FP0"
params = ["Pval<0.05", "Pval<0.01", "Pval<0.005", "Chi>3.84", "Chi>7.68", "Chi>15.36", "TP3_FP0", "TP4_FP0", "TP5_FP0", "All"]


uninonlist = []
runtype = []
for a in params:
    runtype.append([a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

total = []
totalmean = []
totalruns = 0
crossfold = 11

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

            datasetc = pd.read_csv('input/mutfeats.csv', index_col=0)
            datasetc = datasetc[datasetc["class"] != 2]
            datasetc['class'] = datasetc['class'].replace([2],-1)

            if parm[0] == "FP=0":
                sep = df[df["FP"] == 0]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "FP<=1":
                sep = df[df["FP"] <= 1]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "FP<=2":
                sep = df[df["FP"] <= 2]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()

            if parm[0] == "Pval<0.05":
                sep = df[df["p value"] < 0.05]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "Pval<0.01":
                sep = df[df["p value"] < 0.01]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "Pval<0.005":
                sep = df[df["p value"] < 0.005]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()

            if parm[0] == "Chi>3.84":
                sep = df[df["chi"] > 3.84]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "Chi>7.68":
                sep = df[df["chi"] > 7.68]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "Chi>15.36":
                sep = df[df["chi"] > 15.36]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()

            if parm[0] == "TP-2FP>2":
                sep = df[df["tp-fp"] > 2]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "TP-2FP>3":
                sep = df[df["tp-fp"] > 3]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "TP-2FP>4":
                sep = df[df["tp-fp"] > 4]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()

            if parm[0] == "TP3_FP0":
                sep = fp0[fp0["TP"] > 3]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "TP4_FP0":
                sep = fp0[fp0["TP"] > 4]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()
            if parm[0] == "TP5_FP0":
                sep = fp0[fp0["TP"] > 5]
                muts = sep["Feature"].to_list()
                X = datasetc[muts].to_numpy()

            if parm[0] == "Forest":
                ff = pd.read_csv("input/forest.csv")
                muts = ff["Mutation"].to_list()
                X = datasetc[muts].to_numpy()
                
            if parm[0] == "Union":
                muts = list(set(uninonlist))
                X = datasetc[muts].to_numpy()

            if parm[0] == "All":
                muts = df["Feature"].to_list()
                X = datasetc[muts].to_numpy()

            uninonlist += muts

            ############################################################################################

            y = datasetc['class'].to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=(1/crossfold), train_size=1 - (1/crossfold))

            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            clf = RandomForestClassifier(
                oob_score=True, criterion="entropy", n_estimators=100, max_samples=0.7, max_features='sqrt')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            probl = clf.predict_proba(X_test)
            probr = clf.predict_proba(X_test)

            predicted = []
            for x in range(len(y_test)):
                correct = 0
                if y_test[x] == y_pred[x]:
                    correct = 1
                predicted.append([y_test[x], y_test[x],
                                y_pred[x], probl[x][0], probr[x][1], correct])


            # ROC_AUC
            # probs = clf.predict_proba(X_test)
            # probs = probs[:, 1]
            # parm[13] = roc_auc_score(y_test, probs)
            # fpr, tpr, thresholds = roc_curve(y_test, probs)
            # plot_roc_curve(fpr, tpr, cross, parm[0])

            # valuelist.append(parm[13])
            # fprlist.append(fpr)
            # tprlist.append(tpr)

            # # Preciion Recall
            # model = LogisticRegression(solver='lbfgs')
            # model.fit(X_train, y_train)
            # yhat = model.predict_proba(X_test)
            # probs = yhat[:, 1]
            # precision, recall, _ = precision_recall_curve(y_test, probs)
            # parm[14] = auc(recall, precision)
            # plot_pr_curve(y_test, probs, cross, parm[0])

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

            hold = [parm[0], parm[1], parm[2], parm[3], parm[4], parm[5], parm[6], parm[7], parm[8],
                    parm[9], parm[10], parm[11], parm[12], parm[13], parm[14], len(muts)]

            tf = pd.DataFrame(predicted, columns=[
                            "Name", "Actual", "Predicted", "Probaility left", "Probaility Right", "Correctly Classified"])
            tf.to_csv(path+'predicted.csv', index=False)

            features_importance = clf.feature_importances_
            tmp = []
            for i in range(len(muts)):
                tmp.append([muts[i], features_importance[i]])
            tf = pd.DataFrame(tmp, columns=["Mutation", "Value"])
            tf.sort_values(by=['Value'], inplace=True, ascending=False)
            tf.to_csv(path+'feature_importance.csv', index=False)

            print(str(cross + 1), "/", crossfold)
            group.append(hold)
            total.append(hold)

        tf = pd.DataFrame(group, columns=["Seperator Type", "Three-Fold Run", "TP", "FP", "FN", "TN", "Accuracy",
                                        "Sensitivity", "Specificity", "Precision", "Miss Rate", "False discovery rate", "False omission rate", "ROC_AUC", "PR Logistic", "Included Muts"])
        tf.to_csv(patha + 'group_stats.csv', index=False)
        totalruns += 1
        print("\n")
        print(str(totalruns), "/", len(runtype))

        tf = tf.drop("Seperator Type", 1)
        tf = tf.drop("Three-Fold Run", 1)
        tfmean = tf.mean().values.tolist()
        totalmean.append([parm[0]] + tfmean)




        # plt.plot(fprlist[valuelist.index(max(valuelist))], tprlist[valuelist.index(max(valuelist))], color="green", label="Fold:" + str(valuelist.index(max(valuelist))) + "; AUC:" + str(valuelist[valuelist.index(max(valuelist))]))
        # plt.plot(fprlist[np.argsort(valuelist)[len(valuelist)//2]], tprlist[np.argsort(valuelist)[len(valuelist)//2]], color="orange", label="Fold:" + str(np.argsort(valuelist)[len(valuelist)//2])  + "; AUC:" + str(valuelist[np.argsort(valuelist)[len(valuelist)//2]]))
        # plt.plot(fprlist[valuelist.index(min(valuelist))], tprlist[valuelist.index(min(valuelist))], color="red", label="Fold:" + str(valuelist.index(min(valuelist))) + "; AUC:" + str(valuelist[valuelist.index(min(valuelist))]))
        # plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc="lower right")
        # plt.tight_layout()
        # plt.savefig("roc_auc/roc_median/" + str(hold[0]) + ".png")
        # plt.clf()




        print("############################################\n")

    tf = pd.DataFrame(total, columns=["Seperator Type", "Three-Fold Run", "TP", "FP", "FN", "TN", "Accuracy",
                                    "Sensitivity", "Specificity", "Precision", "Miss Rate", "False discovery rate", "False omission rate", "ROC_AUC", "PR Logistic", "Included Muts"])
    tf.to_csv('total_stat.csv', index=False)

    tf = pd.DataFrame(totalmean, columns=["Seperator Type", "TP", "FP", "FN", "TN", "Accuracy", "Sensitivity",
                                        "Specificity", "Precision", "Miss Rate", "False discovery rate", "False omission rate", "ROC_AUC", "PR Logistic",  "Included Muts"])
    tf.to_csv('total_mean.csv', index=False)

    tf = tf[tf["Accuracy"] >= 0.10 - remover]
    
    remover += 0.002
    totalruns = 0
    print(0.9 - remover, "\n\n")
