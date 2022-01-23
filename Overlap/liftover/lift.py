from pyliftover import LiftOver
import pandas as pd

df = pd.read_csv("mutfeats.csv", index_col = 0)
cols = df.columns.tolist()

lo = LiftOver('hg19', 'hg38') # convert from GRCh37 to GRCh38

for x in range(len(cols)):
    if cols[x] != "class":
        #print(cols[x] )
        seg = cols[x].split("_")
        chro = seg[2].split(":")[0]
        start = seg[2].split("-")[0]
        start = start[len(chro)+1:]
        end = seg[2].split("-")[1]
        #print(chro,start,end)
        liftedstart = lo.convert_coordinate("chr"+str(chro),int(start),strand="+")
        liftedend =  lo.convert_coordinate("chr"+str(chro),int(end),strand="+")

        if liftedstart:
            #print(seg)
            #print(liftedstart,liftedend,"\n\n\n")
            seg[1] = "GRCh38"
            seg[2] = chro + ":" + str(liftedstart[0][1]) + "-" + str(liftedend[0][1])

        else:
            seg[1] = "GRCh38"
        df = df.rename(columns={cols[x]:'_'.join(seg)})

df.to_csv("test.csv")