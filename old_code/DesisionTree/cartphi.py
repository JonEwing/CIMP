import csv
import math
import numpy as np
import pandas as pd

def cartphi(featdat):
    cartphi = []
    totalnum = len(featdat.index)
    numfore = len(featdat[featdat['class'] == 1].index)
    numback = len(featdat[featdat['class'] == -1].index)

    counter = 0
    for mut in featdat.columns:
        if mut == 'class':
            continue
        numonright = len(featdat.loc[featdat[mut] == 1.0])
        numonleft = len(featdat.loc[featdat[mut] != 1.0])
        TP = len(featdat.loc[(featdat["class"] == 1.0) & (featdat[mut] == 1.0)])
        FP = len(featdat.loc[(featdat["class"] != 1.0) & (featdat[mut] == 1.0)])
        FN = len(featdat.loc[(featdat["class"] == 1.0) & (featdat[mut] != 1.0)])
        TN = len(featdat.loc[(featdat["class"] != 1.0) & (featdat[mut] != 1.0)])
        
        try:
            phival = (2*(numonleft/totalnum)*(numonright/totalnum)) * (abs((FN/numonleft)-(TP/numonright))+abs((TN/numonleft)-(FP/numonright)))
        except:
            phival = -1

        cartphi.append([mut,phival,TP,FP,FN,TN])
        counter += 1
        if counter % 1000 == 0 :
            print(counter,"/",len(featdat.columns))
    print("\n")
    return(cartphi)

def maxcartphi(infor):
    maxvale = -100
    index = 0
    for x in range(len(infor)):
        if infor[x][1] > maxvale:
            maxvale = infor[x][1]
            index = x
    return index

def nodes(df,info,index):
    left, infol = leftnode(df, info, index)
    right, infor = rightnode(df, info, index)

    infol = cartphi(left)  #Find Left Child
    indexl = maxcartphi(infol)
    print(infol[indexl][0], infol[indexl][1], infol[indexl][2], infol[indexl][3], infol[indexl][4], infol[indexl][5])
    pd.DataFrame(infol).to_csv("left.csv",index = False, header=False)

    infor = cartphi(right) #Find Right Child
    indexr = maxcartphi(infor)
    print(infor[indexr][0], infor[indexr][1], infor[indexr][2], infor[indexr][3], infor[indexr][4], infor[indexr][5])
    pd.DataFrame(infor).to_csv("right.csv",index = False, header=False)

    return left, infol, indexl, right, infor, indexr

def leftnode(left, infol, index):
    count = 0
    for x in left[infol[index][0]]: #Removes all rows that are not in the head node
        if x != 1:
            left = left.drop(left.index[count])
            count -= 1
        count += 1
    left = left.drop(columns= infol[index][0])
    #left = left[left[index == 1]]

    return left, infol

def rightnode(right, infor, index):
    count = 0
    for x in right[infor[index][0]]: #Removes all rows that are in the head node
        if x == 1:
            right = right.drop(right.index[count])
            count -= 1
        count += 1
    right = right.drop(columns= infor[index][0])

    return right, infor



#############################################################################################
featdat = pd.read_csv('mutfeats.csv')
featdat = featdat.drop(columns= "Unnamed: 0")
df = featdat
info = cartphi(featdat) #Left array of cartphi per feature       
index = maxcartphi(info)    #Finds the index of the highest cartphi in an array

print(info[index][0], info[index][1], info[index][2], info[index][3], info[index][4], info[index][5]) #Print Head node


left, infol, indexl, right, infor, indexr = nodes(df,info,index)

cleft, cinfol, cindexl, cright, cinfor, cindexr = nodes(left, infol, indexl)
cleft, cinfol, cindexl, cright, cinfor, cindexr = nodes(right, infor, indexr)

