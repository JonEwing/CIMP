import pandas as pd
df = pd.read_csv('8000.csv',index_col=0)
tf = pd.read_csv('data_clinical_patient.csv',index_col=0)

tf = tf.drop(['#Identifier to uniquely specify a patient.', '#STRING', '#1'], axis=0)
tf = tf['Subtype']

msi = []
for x in df.index:
    hit = False
    for y in tf.index:
        mod_string = x[:len(x) - 3]
        if mod_string == y:
            msi.append(tf.loc[y])
            hit = True
    if hit == False:

        msi.append('N/A')

print(msi)
df['MSI'] = msi
df.to_csv("test.csv")
