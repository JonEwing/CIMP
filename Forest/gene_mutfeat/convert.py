import csv
import math
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

featdat = pd.read_csv('mutfeats_gas.csv', index_col=0)
print("File read")
results = [['Feature', 'TP', 'FP', 'TN', 'FN', 'p value']]

counter = 0
for mut in featdat.columns:
    if mut == 'class':
        continue
    TP = len(featdat.loc[(featdat["class"] == 1.0) & (featdat[mut] == 1.0)])
    FP = len(featdat.loc[(featdat["class"] != 1.0) & (featdat[mut] == 1.0)])
    FN = len(featdat.loc[(featdat["class"] == 1.0) & (featdat[mut] != 1.0)])
    TN = len(featdat.loc[(featdat["class"] != 1.0) & (featdat[mut] != 1.0)])

    conting = pd.DataFrame([[TP, FP], [FN, TN]], columns=[
                           "cimp-h", 'non-cimp-h'])

    results.append([mut, TP, FP, TN, FN, fisher_exact(conting)[1]])

    if counter % 1000 == 0:
        print(str(counter) + " / " + str(len(featdat.columns)))

    counter += 1

tf = pd.DataFrame(results)
tf.to_csv('new_results.csv', index=False, header=False)
