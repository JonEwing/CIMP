import pandas as pd

df = pd.read_csv("mutfeats.csv")

counter = 0
cimpcount = 1
noncimpcount = 1

for i in df['class']:
    if i == 1:
        df.loc[counter, "Unnamed: 0"] = "C" + str(cimpcount)
        cimpcount += 1
    else:
        df.loc[counter, "Unnamed: 0"] = "NC" + str(noncimpcount)
        noncimpcount += 1
    counter += 1

df = df.rename(columns = {"Unnamed: 0" : ""})
df.to_csv('junk.csv', index=False)