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

# ------- Feature Engineering and Encoding ------- # 
def encode_applicationType(data):
    data['application_type'] = np.where(data['application_type']=='INDIVIDUAL', 0, data['application_type'])
    data['application_type'] = np.where(data['application_type']=='JOINT', 1, data['application_type'])
    data['application_type'] = data['application_type'].astype(float)
    return data

def encode_listStatus(data):
    data['initial_list_status'] = np.where(data['initial_list_status']=='f', 0, data['initial_list_status'])
    data['initial_list_status'] = np.where(data['initial_list_status']=='w', 1, data['initial_list_status'])
    data['initial_list_status'] = data['initial_list_status'].astype(float)
    return data

def encode_term(data):
    data['term'] = np.where(data['term']==' 36 months', 0, data['term'])
    data['term'] = np.where(data['term']==' 60 months', 1, data['term'])
    data['term']=data['term'].astype(float)
    return data

def encode_grade(data):
    data['grade'] = np.where(data['grade']=='A', 0, data['grade'])
    data['grade'] = np.where(data['grade']=='B', 0, data['grade'])
    data['grade'] = np.where(data['grade']=='C', 0, data['grade'])
    data['grade'] = np.where(data['grade']=='D', 1, data['grade'])
    data['grade'] = np.where(data['grade']=='E', 1, data['grade'])
    data['grade'] = np.where(data['grade']=='F', 1, data['grade'])
    data['grade'] = np.where(data['grade']=='G', 1, data['grade'])
    data['grade'] = data['grade'].astype(float)
    return data

def encode_homeOwnership(data):
    data['home_ownership'] = np.where(data['home_ownership']=='RENT', 1, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='OWN', 1, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='MORTGAGE', 1, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='NONE', 2, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='OTHER', 2, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='ANY', 0, data['home_ownership'])
    data['home_ownership'] = data['home_ownership'].astype(float)
    return data

def convert_lastCreditPullD(data):
    data['last_credit_pull_d'] = pd.to_datetime(data['last_credit_pull_d'])
    data['Month'] = data['last_credit_pull_d'].apply(lambda x: x.month)
    data['Year'] = data['last_credit_pull_d'].apply(lambda x: x.year)
    data = data.drop(['last_credit_pull_d'], axis = 1)
    return data

def encode_all(data):
    data = encode_listStatus(data)
    data = encode_applicationType(data)
    data = encode_term(data)
    data = encode_grade(data)
    data = encode_homeOwnership(data)
    data = convert_lastCreditPullD(data)
    return data

# -------- Execution / main --------- #
data = txt_to_dataframe(path)
data = delete_col(data,800000)
data = remove_lowCor(data,0.02)
print(data.shape)
data = encode_all(data)
print(data.info())