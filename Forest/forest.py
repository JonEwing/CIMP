import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

dataset = pd.read_csv('../data/mutfeats no cimp class labels.csv',index_col=0)

occur4 = []
todata = []
for mut in dataset.columns:
    TP = 0
    FP = 0
    for sample in dataset.index:
        if dataset.loc[sample,mut] == 1 and sample[0] == 'C':
            TP += 1
        if dataset.loc[sample,mut] == 1 and sample[0] != 'C':
            FP += 1
    if TP + FP >= 5:
        occur4.append(mut)

print(len(occur4))
dataset = dataset[occur4]

print("Build Data Finish")
############################################################################################

classlist = []
for x in dataset.index:
    if x[0] == 'C':
        classlist.append(1)
    else:
        classlist.append(0)

dataset['class'] = classlist

y = dataset['class']
X = dataset.drop('class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, train_size = 0.66)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


clf = RandomForestClassifier(n_estimators = 50000, max_samples = .9, oob_score = True, criterion = "entropy")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
probl = clf.predict_proba(X_test)
probr = clf.predict_proba(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))


predicted = []
for x in range(len(y_test.index)):
    correct = 0
    if y_test[x] == y_pred[x]:
        correct = 1
    predicted.append([y_test.index[x], y_test[x], y_pred[x], probl[x][0], probr[x][1], correct])

tf = pd.DataFrame(predicted, columns = ["Name", "Actual", "Predicted", "Probaility left", "Probaility Right", "Correctly Classified"])
tf.to_csv('predicted.csv', index=False)

sel = SelectFromModel(clf)
sel.fit(X_train, y_train)
selected_feat= X.columns[(sel.get_support())]

f = open("important_mut.txt", "w")
f.write("Most important features:\n")
for x in selected_feat:
    f.write("\n")
    f.write(x)
f.close()