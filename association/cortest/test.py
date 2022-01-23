import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

preset = []
for a in range(7, 1, -1):
    df = pd.read_csv('rules_ap.csv', sep=',')
    tf = df[df["Total_len"] == a]
    test = tf["antecedents"].to_list()
    test += tf["consequents"].to_list()

    totalset = []
    for x in test:
        temp = x.split(",")
        for y in temp:
            y = y.replace("{","")
            y = y.replace("}","")
            y = y.replace('"',"")
            y = y.replace("'","")
            y = y.replace(" ","")
            totalset.append(y)

    if x != 7:
        totalset = list(set(totalset))
        for i in preset:
            try:
                totalset.remove(i)
            except:
                continue
        preset += totalset
        print("\n\nMutation number:",a, "/ Total Length:", len(totalset))
        for x in totalset:
            print(x)
    else:
        totalset = list(set(totalset))
        print("\n\nMutation number:",a, "/ Total Length:", len(totalset))
        for x in totalset:
            print(x)
        preset = totalset

"""

df = pd.read_csv('mutfeats.csv', index_col=0)
setlist = ["DRD5_GRCh37_4:9785421-9785421_3'UTR_SNP_G-G-T", 'PRRG1_GRCh37_X:37312611-37312611_Frame-Shift-Del_DEL_C-C--', 'PTENP1_GRCh37_9:33674810-33674810_RNA_SNP_A-A-G', "DDHD1_GRCh37_14:53513450-53513480_3'UTR_DEL_AATCAGTTTTAGGCCATTCATGTCCTTCAAG-AATCAGTTTTAGGCCATTCATGTCCTTCAAG-TCAGTTTTAGGCCATTCATGTCCTTCA", "NSD1_GRCh37_5:176722543-176722543_3'UTR_DEL_A-A--", "TNFSF11_GRCh37_13:43181715-43181715_3'UTR_DEL_A-A--", "RPL22_GRCh37_1:6257785-6257785_Frame-Shift-Del_DEL_T-T--"]

tf = df[setlist]

tf = tf[(tf.T != 0).any()]
tf.rename(columns={"DDHD1_GRCh37_14:53513450-53513480_3'UTR_DEL_AATCAGTTTTAGGCCATTCATGTCCTTCAAG-AATCAGTTTTAGGCCATTCATGTCCTTCAAG-TCAGTTTTAGGCCATTCATGTCCTTCA":"DDHD1_GRCh37_14:53513450-53513480_3'UTR_DEL_AATCA..."}, inplace=True)

bigrows = df.index.to_list()
rows = tf.index.to_list()
classes = df["class"].to_list()
classlist = {}
for x in range(len(bigrows)):
    for y in rows:
        if y == bigrows[x]:
            if classes[x] == -1:
                classlist[y] = 'red'
            elif classes[x] == 1:
                classlist[y] = 'green'
            else:
                classlist[y] = 'darkorange'

af = pd.read_csv('msi.csv', sep=',', index_col=0)

bigrows = df.index.to_list()
rows = tf.index.to_list()
classes = af["MSI"].to_list()
classlist = {}
for x in range(len(bigrows)):
    for y in rows:
        if y == bigrows[x]:
            if classes[x] == "MSI: pMSS":
                classlist[y] = 'red'
            elif classes[x] == 'MSI: MSI-H':
                classlist[y] = 'darkgreen'
            elif classes[x] == 'MSI: pMSI':
                classlist[y] = 'limegreen'
            elif classes[x] == 'MSI: NC':
                classlist[y] = 'black'


tf = pd.read_csv('data.csv', sep=',', index_col=0)

ax = sns.clustermap(tf, figsize=(15,15), yticklabels=True, cmap="YlGnBu")
for tick_label in ax.ax_heatmap.axes.get_yticklabels():
    ttxt = tick_label.get_text()
    tick_label.set_color(classlist[ttxt])

plt.setp(ax.ax_heatmap.xaxis.get_majorticklabels(), rotation=-10)
plt.savefig('test.png')
"""