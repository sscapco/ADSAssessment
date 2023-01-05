import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "C:\\Users\\qsrt\OneDrive - Capco\\Documents\\ADS\\xyz\\data\\XYZCorp_LendingData.txt"

def txt_to_dataframe(path):
    data = pd.read_table(path,parse_dates=['issue_d'],low_memory=False)
    return data

def delete_col(data,N):
    colRec=data.isnull().sum()
    for i in range(len(colRec)):
        if colRec[i]>N:
            del data['{}'.format(colRec.index[i])]
    return data

def remove_lowCor(data,n):
    corCol = data[data.columns].corr()['default_ind'][:]
    corCol2=[]
    for i in range(len(corCol)):
        if corCol[i]<n and corCol[i]>(-1*n):
            corCol2.append(corCol.index[i])
    for i in range(len(corCol2)):
        del data['{}'.format(corCol2[i])]
    return data

data = txt_to_dataframe(path)
data = delete_col(data,800000)
data = remove_lowCor(data,0.02)
print(data.shape)