import pandas as pd
from pyvis.network import Network
from networkx.drawing.nx_agraph import graphviz_layout
#import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx

sources = []
targets = []
weights = []

df = pd.read_csv('mutfeats.csv', index_col=0)
setlist = ["DRD5_GRCh37_4:9785421-9785421_3'UTR_SNP_G-G-T", 'PRRG1_GRCh37_X:37312611-37312611_Frame-Shift-Del_DEL_C-C--', 'PTENP1_GRCh37_9:33674810-33674810_RNA_SNP_A-A-G', "DDHD1_GRCh37_14:53513450-53513480_3'UTR_DEL_AATCAGTTTTAGGCCATTCATGTCCTTCAAG-AATCAGTTTTAGGCCATTCATGTCCTTCAAG-TCAGTTTTAGGCCATTCATGTCCTTCA", "NSD1_GRCh37_5:176722543-176722543_3'UTR_DEL_A-A--", "TNFSF11_GRCh37_13:43181715-43181715_3'UTR_DEL_A-A--", "RPL22_GRCh37_1:6257785-6257785_Frame-Shift-Del_DEL_T-T--"]

tf = df[setlist]

tf = tf[(tf.T != 0).any()]

tf.to_csv("save.csv")
listlist = []
for x in range(len(setlist)):
    for y in range(len(setlist)):
        if y > x:
            af = tf[setlist[x]]
            bf = tf[setlist[y]]
            
            test = ((tf[setlist[x]] == 1) & (tf[setlist[y]] == 1)).value_counts()
            if test[1] > 5:
                sources.append(setlist[x].split("_")[0])
                targets.append(setlist[y].split("_")[0])
                weights.append(test[1])

got_net = Network(height='100%', width='100%')
got_net.barnes_hut()

edge_data = zip(sources, targets, weights)

for e in edge_data:
    src = e[0]
    dst = e[1]
    w = str(e[2])

    got_net.add_node(src, src, title=src)
    got_net.add_node(dst, dst, title=dst)
    got_net.add_edge(src, dst, value=w, label = w)

neighbor_map = got_net.get_adj_list()

got_net.show('gameofthrones.html')