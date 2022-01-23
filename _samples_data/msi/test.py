import pandas as pd
df = pd.read_csv('300.csv',index_col=0)

tf = pd.read_csv('datafile.S1.1.KeyClinicalData.csv',index_col=0)
tf = tf['msi_status_7_marker_call']

msiarray = ["RPL22_GRCh37_1:6257785-6257785_Frame-Shift-Del_DEL_T-T--","RNF43_GRCh37_17:56435161-56435161_Frame-Shift-Del_DEL_C-C--","KRAS_GRCh37_12:25398284-25398284_Missense-Mutation_SNP_C-C-A_C-C-T_C-C-G","DOCK3_GRCh37_3:51417604-51417604_Frame-Shift-Del_DEL_C-C--","ACVR2A_GRCh37_2:148683686-148683686_Frame-Shift-Del_DEL_A-A--","SHC1_GRCh37_1:154935863-154935863_3'UTR_DEL_G-G--","FAM46A_GRCh37_6:82457993-82457993_3'UTR_DEL_T-T--","KIAA1024_GRCh37_15:79750586-79750586_Frame-Shift-Del_DEL_A-A--","SETD1B_GRCh37_12:122242658-122242658_Frame-Shift-Del_DEL_C-C--","ATPAF1_GRCh37_1:47100598-47100598_3'UTR_DEL_A-A--"]
mssarray = ["KRAS_GRCh37_12:25398284-25398284_Missense-Mutation_SNP_C-C-A_C-C-T_C-C-G","PTEN_GRCh37_10:89692904-89692904_Missense-Mutation_SNP_C-C-G","BCOR_GRCh37_X:39921444-39921444_Missense-Mutation_SNP_T-T-C","KRAS_GRCh37_12:25398285-25398285_Missense-Mutation_SNP_C-C-A_C-C-T","PLAG1_GRCh37_8:57077122-57077122_3'UTR_DEL_T-T--","ARL2BP_GRCh37_16:57286855-57286855_3'UTR_DEL_A-A--","ARID1A_GRCh37_1:27105930-27105931_Frame-Shift-Ins_INS_----G","KMT2C_GRCh37_7:151833642-151833642_3'UTR_DEL_A-A--","GRHL3_GRCh37_1:24681723-24681723_Intron_DEL_T-T--","BCL11B_GRCh37_14:99638058-99638058_3'UTR_DEL_T-T--"]
msi = []
for x in df.index:
    hit = False
    for y in tf.index:
        mod_string = x[:len(x) - 3]
        if mod_string == y:
            msi.append("MSI: "+str(tf.loc[y]))
            hit = True
    if hit == False:
        msinum = 0
        mssnum = 0
        for z in range(len(msiarray)):
            if df.loc[x,msiarray[z]] == 1:
                msinum += 1
            if df.loc[x,mssarray[z]] == 1:
                mssnum += 1

        if msinum > mssnum:
            msi.append('MSI: pMSI')
        elif mssnum > msinum:
            msi.append('MSI: pMSS')
        else:
            msi.append('MSI: NC')

df.insert(0, 'MSI', msi)

#########################################################################################

msiarraycart = ["RBFOX1_GRCh37_16:7762402-7762402_3'Flank_DEL_A-A--","TFAP2B_GRCh37_6:50814901-50814901_3'UTR_DEL_A-A--","LATS2_GRCh37_13:21547831-21547831_3'UTR_DEL_A-A--","MGAT3_GRCh37_22:39888064-39888064_3'UTR_DEL_T-T--","BHLHE40_GRCh37_3:5026719-5026719_3'UTR_DEL_T-T--","C18orf25_GRCh37_18:43844099-43844099_3'UTR_DEL_A-A--","PIK3R2_GRCh37_19:18273784-18273784_Missense-Mutation_SNP_G-G-A","ITGAV_GRCh37_2:187542366-187542367_3'UTR_INS_----TTG","NFASC_GRCh37_1:204924033-204924033_Frame-Shift-Del_DEL_C-C--","SENP2_GRCh37_3:185347892-185347892_3'UTR_DEL_T-T--"]
mssarraycart = msiarraycart

msi = []
for x in df.index:
    hit = False
    for y in tf.index:
        mod_string = x[:len(x) - 3]
        if mod_string == y:
            msi.append("MSICartPhi: "+str(tf.loc[y]))
            hit = True
    if hit == False:
        msinum = 0
        mssnum = 0
        for z in range(len(msiarraycart)):
            if df.loc[x,msiarraycart[z]] == 1:
                msinum += 1
        for z in range(len(mssarraycart)):
            if df.loc[x,mssarraycart[z]] == 1:
                mssnum += 1

        if msinum > mssnum:
            msi.append('MSICartPhi: pMSI')
        elif mssnum > msinum:
            msi.append('MSICartPhi: pMSS')
        else:
            msi.append('MSICartPhi: NC')   

df.insert(0, 'MSICartPhi', msi)

#########################################################################################

msiarrayinfo = ["RBFOX1_GRCh37_16:7762402-7762402_3'Flank_DEL_A-A--","TFAP2B_GRCh37_6:50814901-50814901_3'UTR_DEL_A-A--","LATS2_GRCh37_13:21547831-21547831_3'UTR_DEL_A-A--","MGAT3_GRCh37_22:39888064-39888064_3'UTR_DEL_T-T--","BHLHE40_GRCh37_3:5026719-5026719_3'UTR_DEL_T-T--","C18orf25_GRCh37_18:43844099-43844099_3'UTR_DEL_A-A--","PIK3R2_GRCh37_19:18273784-18273784_Missense-Mutation_SNP_G-G-A","ITGAV_GRCh37_2:187542366-187542367_3'UTR_INS_----TTG","NFASC_GRCh37_1:204924033-204924033_Frame-Shift-Del_DEL_C-C--","SENP2_GRCh37_3:185347892-185347892_3'UTR_DEL_T-T--"]
mssarrayinfo = ["PLAG1_GRCh37_8:57077122-57077122_3'UTR_DEL_T-T--","ARL2BP_GRCh37_16:57286855-57286855_3'UTR_DEL_A-A--","ARID1A_GRCh37_1:27105930-27105931_Frame-Shift-Ins_INS_----G","KMT2C_GRCh37_7:151833642-151833642_3'UTR_DEL_A-A--","GRHL3_GRCh37_1:24681723-24681723_Intron_DEL_T-T--","BCL11B_GRCh37_14:99638058-99638058_3'UTR_DEL_T-T--","EBF1_GRCh37_5:158125830-158125830_3'UTR_DEL_A-A--","DLG3_GRCh37_X:69722275-69722275_3'UTR_DEL_T-T--","KIAA0355_GRCh37_19:34844830-34844830_3'UTR_DEL_A-A--","DYRK1A_GRCh37_21:38853058-38853059_Frame-Shift-Ins_INS_----A"]

msi = []
for x in df.index:
    hit = False
    for y in tf.index:
        mod_string = x[:len(x) - 3]
        if mod_string == y:
            msi.append("MSIInfoGain: "+str(tf.loc[y]))
            hit = True
    if hit == False:
        msinum = 0
        mssnum = 0
        for z in range(len(msiarrayinfo)):
            if df.loc[x,msiarrayinfo[z]] == 1:
                msinum += 1
        for z in range(len(mssarrayinfo)):
            if df.loc[x,mssarrayinfo[z]] == 1:
                mssnum += 1

        if msinum > mssnum:
            msi.append('MSIInfoGain: pMSI')
        elif mssnum > msinum:
            msi.append('MSIInfoGain: pMSS')
        else:
            msi.append('MSIInfoGain: NC')   

df.insert(0, 'MSIInfoGain', msi)

#############################################################################################

tf = pd.read_csv('datafile.S1.1.KeyClinicalData.csv',index_col=0)
tf = tf['IntegrativeCluster']

Subtype = []
for x in df.index:
    hit = False
    for y in tf.index:
        mod_string = x[:len(x) - 3]
        if mod_string == y:
            Subtype.append("Subtype: "+str(tf.loc[y]))
            hit = True
        if mod_string == 'nan':
            print("nan")
    if hit == False:
        Subtype.append('Subtype: N/A')


df.insert(0, 'Subtype', Subtype)

########################################################################################

tf = pd.read_csv('totalclust.csv',index_col=0)
tf = tf['Cluster']

Cluster = []

for x in df.index:
    hit = False
    for y in tf.index:
        if x == y:
            Cluster.append("Cluster: "+str(tf.loc[y]))
            hit = True
    if hit == False:
        Cluster.append('Cluster: N/A')

df.insert(0, 'Cluster', Cluster)

namedrows = []
for x in df.sum(axis=1):
    namedrows.append("Sum: " + str(x))
df.insert(0, 'Sum', namedrows)

control = [1] * len(namedrows)
df['Control'] = [1] * len(namedrows)

df.to_csv("test.csv")


cf = pd.read_csv('results.csv', sep=',')

cf['chi'] = (((cf["TP"] - cf["TP"].mean()) ** 2) / cf["TP"].mean()) + (((cf["FP"] - cf["FP"].mean()) ** 2) / cf["FP"].mean()) + \
    (((cf["TN"] - cf["TN"].mean()) ** 2) / cf["TN"].mean()) + \
    (((cf["FN"] - cf["FN"].mean()) ** 2) / cf["FN"].mean())

sep = cf[df["chi"] > 15]
totalmuts = sep["Feature"].to_list()

df = df.filter(totalmuts)
df.to_csv("test2.csv")
