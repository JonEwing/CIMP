import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig

df = pd.read_csv('results.csv', sep=',')

df['chi'] = (((df["TP"] - df["TP"].mean()) ** 2) / df["TP"].mean()) + (((df["FP"] - df["FP"].mean()) ** 2) / df["FP"].mean()) + \
    (((df["TN"] - df["TN"].mean()) ** 2) / df["TN"].mean()) + \
    (((df["FN"] - df["FN"].mean()) ** 2) / df["FN"].mean())

sep = df[df["chi"] > 15]
totalmuts = sep["Feature"].to_list()

#totalmuts = pd.read_csv('mut_list.csv', sep=',')
#totalmuts = totalmuts["Mutations"].to_list()

df = pd.read_csv('mutfeats.csv', sep=',', index_col = 0)

df = df.drop("class", axis = 1)
df = df.filter(totalmuts)

corrMatrix = df.corr()

plt.subplots(figsize=(30,25))
sn.color_palette("tab10")
svm = sn.heatmap(corrMatrix)
figure = svm.get_figure()    
figure.savefig('svm_conf.png')