import pandas as pd

genes = pd.read_csv('data/gene_list.txt', header = None)
genes = genes[0].tolist() 

cimp = pd.read_csv('data/cimp_classes.csv', header = None)
cimp_samples = cimp[0].tolist()
cimp_classes = cimp[2].tolist()

muts = pd.read_csv('data/mutfeats.csv', low_memory=False)
mut_samples = muts['Unnamed: 0'].tolist()

expression_data = pd.read_csv('data/UCEC.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt',  sep="\t", low_memory=False)
expression_genes = expression_data["Hybridization REF"].tolist()
expression_samples = expression_data.columns

expression_data2 = pd.read_csv('data/UCS.rnaseqv2__illuminahiseq_rnaseqv2__unc_edu__Level_3__RSEM_genes_normalized__data.data.txt',  sep="\t", low_memory=False)
expression_samples2 = expression_data2.columns

hold = []
row_names = []
for x in expression_genes:
    for y in genes:
        if x.split("|")[0] == y:
            hold.append(x)
            row_names.append(y)
expression_data = expression_data[expression_data['Hybridization REF'].isin(hold)]
expression_data2 = expression_data2[expression_data2['Hybridization REF'].isin(hold)]

hold = []
col_names = []
hold.append("Hybridization REF")
for x in expression_samples:
    for y in mut_samples:
        if y in x:
            hold.append(x)
            col_names.append(y)
expression_data = expression_data[hold]

hold2 = []
col_names2 = []
hold.append("Hybridization REF")
for x in expression_samples2:
    for y in mut_samples:
        if y in x:
            hold2.append(x)
            col_names2.append(y)
expression_data2 = expression_data2[hold2]

hold = hold + hold2
expression_data = expression_data.join(expression_data2)

cimpclass = []
cimpclass.append("Class")
for x in hold:
    for y in range(len(cimp_samples)):
        if x[:len(x)-13] == cimp_samples[y]:
            if cimp_classes[y] == "CIMP+":
                cimpclass.append(1)
            elif cimp_classes[y] == "CIMP-":
                cimpclass.append(0)
            elif cimp_classes[y] == "CIMPi":
                cimpclass.append(2)
            else:
                cimpclass.append(3)



expression_data = expression_data.T
expression_data['Class'] = cimpclass
expression_data.to_csv("data/raw_data.csv", header=False)