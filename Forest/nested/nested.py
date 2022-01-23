import pandas as pd
from numpy import mean
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import random

def plot_roc_curve(fpr, tpr, Run, params, roc):
	plt.plot(fpr, tpr, color=[random.random(), random.random(), random.random()], label="Fold:" + str(Run)+ " ROC:"+ str(round(roc * 100,1)) + " " + str(params))
	plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend(loc="lower center", bbox_to_anchor=(0.5, -.7))
	plt.tight_layout()


def stats(y_test, y_pred):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for x in range(len(y_test)):
		if y_test[x] == 1 and y_pred[x] == 1:
			TP += 1
		if y_test[x] != 1 and y_pred[x] == 1:
			FP += 1
		if y_test[x] != 1 and y_pred[x] != 1:
			TN += 1
		if y_test[x] == 1 and y_pred[x] != 1:
			FN += 1
		
		accu = (TP + TN)/ (TP + FP + TN + FN)
		try:
			Sensitivity = (TP) / (TP + FN)
		except:
			Sensitivity = -1
		try:
			Specificity = (TN) / (TN + FP)
		except:
			Specificity = -1
		try:
			Precision = (TP) / (TP + FP)
		except:
			Precision = -1
		try:
			Missrate = (FN) / (FN + TP)
		except:
			Missrate = -1
		try:
			fdr = (FP) / (FP + TP)
		except:
			fdr = -1
		try:
			fomr = (FN) / (FN + TN)
		except:
			fomr = -1
	return [TP, FP, FN, TN, accu, Sensitivity, Specificity, Precision, Missrate, fdr, fomr]

outerloop = 10
interloop = 10
meanhold = []
hold = []
uninonlist = []
df = pd.read_csv("input/results.csv")

df["a/e"] = (df["TP"] / (df["TP"] + df["FN"])) / (df["FP"] / (df["FP"] + df["TN"]))
df['chi'] = (((df["TP"] - df["TP"].mean()) ** 2) / df["TP"].mean()) + (((df["FP"] - df["FP"].mean()) ** 2) / df["FP"].mean()) + (((df["TN"] - df["TN"].mean()) ** 2) / df["TN"].mean()) + (((df["FN"] - df["FN"].mean()) ** 2) / df["FN"].mean())
df["tp-fp"] = df["TP"] - 2*df["FP"]
fp0 = df[df['FP'] == 0]

datasetc = pd.read_csv('input/mutfeats.csv',index_col=0)
datasetc = datasetc[datasetc["class"] != 2]

#"AE>10", "AE>7.5", "AE>5","FP=0", "FP<=1", "FP<=2", "Pval<0.05", "Pval<0.01", "Pval<0.005", "Chi>3.84", "Chi>7.68", "Chi>15.36", "TP-2FP>2", "TP-2FP>3", "TP-2FP>4", "TP3_FP0", "TP4_FP0", "TP5_FP0", "Forest", "Union", "All"
params = ["AE>10", "AE>7.5", "AE>5","FP=0", "FP<=1", "FP<=2", "Pval<0.05", "Pval<0.01", "Pval<0.005", "Chi>3.84", "Chi>7.68", "Chi>15.36", "TP-2FP>2", "TP-2FP>3", "TP-2FP>4", "TP3_FP0", "TP4_FP0", "TP5_FP0", "Forest", "Union", "All"]
for parm in params:
	plt.figure(figsize=(8,7))
	if parm == "AE>10":
		sep = df[df["a/e"].between(10,10000)]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "AE>7.5":
		sep = df[df["a/e"].between(7.5,10000)]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "AE>5":
		sep = df[df["a/e"].between(5,10000)]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()

	if parm == "FP=0":
		sep = df[df["FP"] == 0]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "FP<=1":
		sep = df[df["FP"] <= 1]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "FP<=2":
		sep = df[df["FP"] <= 2]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()

	if parm == "Pval<0.05":
		sep = df[df["p value"] < 0.05]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "Pval<0.01":
		sep = df[df["p value"] < 0.01]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "Pval<0.005":
		sep = df[df["p value"] < 0.005]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()

	if parm == "Chi>3.84":
		sep = df[df["chi"] > 3.84]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "Chi>7.68":
		sep = df[df["chi"] > 7.68]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "Chi>15.36":
		sep = df[df["chi"] > 15.36]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()

	if parm == "TP-2FP>2":
		sep = df[df["tp-fp"] > 2]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "TP-2FP>3":
		sep = df[df["tp-fp"] > 3]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "TP-2FP>4":
		sep = df[df["tp-fp"] > 4]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()

	if parm == "TP3_FP0":
		sep = fp0[fp0["TP"] > 3]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "TP4_FP0":
		sep = fp0[fp0["TP"] > 4]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()
	if parm == "TP5_FP0":
		sep = fp0[fp0["TP"] > 5]
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()

	if parm == "Forest":
		forst = pd.read_csv("input/combined_muts.csv")
		muts = forst["Mutations"].to_list()
		X = datasetc[muts].to_numpy()

	if parm == "Union":
		muts = list(set(uninonlist))
		X = datasetc[muts].to_numpy()

	if parm == "All":
		muts = sep["Feature"].to_list()
		X = datasetc[muts].to_numpy()

	uninonlist += muts

	y = datasetc['class'].to_numpy()
	cv_outer = KFold(n_splits = outerloop, shuffle=True)
	outer_results = list()
	counter = 0
	average = []
	for train_ix, test_ix in cv_outer.split(X):
		X_train, X_test = X[train_ix, :], X[test_ix, :]
		y_train, y_test = y[train_ix], y[test_ix]
		cv_inner = KFold(n_splits= interloop, shuffle=True)
		model = RandomForestClassifier(oob_score = True, criterion = "entropy", n_estimators = 100, max_samples = 0.7, max_features = 'sprt')

		space = dict()
		space[''] = [50, 100, 1000]
		space['max_features'] = ["sqrt", "log2"]
		space['max_samples'] = [0.65, 0.75]

		search = GridSearchCV(model, space, scoring='roc_auc', cv=cv_inner, refit=True)
		result = search.fit(X_train, y_train)

		best_model = result.best_estimator_
		yhat = best_model.predict(X_test)
		acc = accuracy_score(y_test, yhat)
		outer_results.append(acc)
		
		probs = best_model.predict_proba(X_test)
		probs = probs[:, 1]
		roc = roc_auc_score(y_test, probs)
		fpr, tpr, thresholds = roc_curve(y_test, probs)
		plot_roc_curve(fpr, tpr, counter, result.best_params_, roc)

		statsitcs = stats(y_test, yhat)
		statsitcs = statsitcs + [roc]
		statsitcs = statsitcs[0:4] + [result.best_score_] + statsitcs[4:]
		hold.append([parm, result.best_params_, len(muts), counter] + statsitcs + [fpr] + [tpr] + [thresholds])
		average.append(statsitcs)
		avgs = pd.DataFrame(average)
		
		print('>acc=%.3f, est=%.3f, cfg=%s, roc_auc=%.3f' % (acc, result.best_score_, result.best_params_, roc))
		counter += 1

	plt.savefig("roc_auc/" + parm + ".png")
	plt.clf()
	meanvals = avgs.mean().to_list()
	meanhold.append([parm] + [len(muts)] + meanvals)
	print('%s :Accuracy: %.3f\n###############################################\n' % (parm, mean(outer_results)))

tf = pd.DataFrame(hold, columns = ["Seperator Type", "parameters", "Mutation Amount", "Cross Fold", "TP", "FP", "FN", "TN","Accuracy Test set", "Accuracy Validation Set", "Sensitivity", "Specificity", "Precision", "Miss Rate", "False discovery rate", "False omission rate", "roc_auc", "fpr", "tpr", "thresholds"])
tf.to_csv('innerfolds.csv', index=False)
tf = pd.DataFrame(meanhold, columns = ["Seperator Type", "Mutation Amount", "TP", "FP", "FN", "TN","Accuracy Test set", "Accuracy Validation Set", "Sensitivity", "Specificity", "Precision", "Miss Rate", "False discovery rate", "False omission rate", "roc_auc"])
tf.to_csv('outerfolds.csv', index=False)
