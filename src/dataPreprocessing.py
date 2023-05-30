import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def txt_to_dataframe(path):
    """
    Converts txt file into pandas dataframe.
    -------
    Parameter
    path: path to txt dataset.
    --------
    Returns
    data: pandas dataframe of data.
    """
    data = pd.read_table(path,parse_dates=['issue_d'],low_memory=False)
    return data

def delete_col(data,N):
    """
    Deletes column with too many null values
    -------
    Parameter
    data: pandas dataframe.
    N: number of nulls threshold to delete column
    --------
    Returns
    data: pandas datafram.
    """
    colRec=data.isnull().sum()
    for i in range(len(colRec)):
        if colRec[i]>N:
            del data['{}'.format(colRec.index[i])]
    return data

def remove_lowCor(data,n,col='default_ind'):
    """
    Deletes column with low correlation to specified column
    -------
    Parameter
    data: pandas dataframe.
    n: correlation threshold.
    col: column name.
    --------
    Returns
    data: pandas dataframe.
    """
    corCol = data[data.columns].corr()[col][:]
    corCol2=[]
    for i in range(len(corCol)):
        if corCol[i]<n and corCol[i]>(-1*n):
            corCol2.append(corCol.index[i])
    for i in range(len(corCol2)):
        del data['{}'.format(corCol2[i])]
    return data

def delete_objType(data):
    colObj=[]
    for i in range(len(data.dtypes)):
        if data.dtypes[i]!='object':
            colObj.append(data.dtypes.index[i])
    data1= data[colObj]
    return data1

def delete_extraCols(data):
    """
    Deletes specific columns 
    -------
    Parameter
    data: pandas dataframe.
    --------
    Returns
    data: pandas dataframe.
    """
    del data['mths_since_last_record']
    del data['id']
    del data['member_id']
    del data['tot_cur_bal']
    del data['total_rev_hi_lim']
    del data['policy_code']
    del data['issue_d']
    return data

# ------- Feature Engineering and Encoding ------- # 
def encode_applicationType(data):
    """
    Encode columns 
    -------
    Parameter
    data: pandas dataframe.
    --------
    Returns
    data: pandas dataframe.
    """
    data['application_type'] = np.where(data['application_type']=='INDIVIDUAL', 0, data['application_type'])
    data['application_type'] = np.where(data['application_type']=='JOINT', 1, data['application_type'])
    data['application_type'] = data['application_type'].astype(float)
    return data

def encode_listStatus(data):
    """
    Encode columns 
    -------
    Parameter
    data: pandas dataframe.
    --------
    Returns
    data: pandas dataframe.
    """
    data['initial_list_status'] = np.where(data['initial_list_status']=='f', 0, data['initial_list_status'])
    data['initial_list_status'] = np.where(data['initial_list_status']=='w', 1, data['initial_list_status'])
    data['initial_list_status'] = data['initial_list_status'].astype(float)
    return data

def encode_term(data):
    """
    Encode columns 
    -------
    Parameter
    data: pandas dataframe.
    --------
    Returns
    data: pandas dataframe.
    """
    data['term'] = np.where(data['term']==' 36 months', 0, data['term'])
    data['term'] = np.where(data['term']==' 60 months', 1, data['term'])
    data['term']=data['term'].astype(float)
    return data

def encode_grade(data):
    """
    Encode columns 
    -------
    Parameter
    data: pandas dataframe.
    --------
    Returns
    data: pandas dataframe.
    """
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
    """
    Encode columns 
    -------
    Parameter
    data: pandas dataframe.
    --------
    Returns
    data: pandas dataframe.
    """
    data['home_ownership'] = np.where(data['home_ownership']=='RENT', 1, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='OWN', 1, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='MORTGAGE', 1, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='NONE', 2, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='OTHER', 2, data['home_ownership'])
    data['home_ownership'] = np.where(data['home_ownership']=='ANY', 0, data['home_ownership'])
    data['home_ownership'] = data['home_ownership'].astype(float)
    return data

def convert_lastCreditPullD(data):
    """
    Encode date column to get month and year. 
    -------
    Parameter
    data: pandas dataframe.
    --------
    Returns
    data: pandas dataframe.
    """
    data['last_credit_pull_d'] = pd.to_datetime(data['last_credit_pull_d'])
    data['Month'] = data['last_credit_pull_d'].apply(lambda x: x.month)
    data['Year'] = data['last_credit_pull_d'].apply(lambda x: x.year)
    data = data.drop(['last_credit_pull_d'], axis = 1)
    return data

def encode_all(data):
    """
    Encode columns using all encoding functions 
    -------
    Parameter
    data: pandas dataframe.
    --------
    Returns
    data: pandas dataframe.
    """
    data = encode_listStatus(data)
    data = encode_applicationType(data)
    data = encode_term(data)
    data = encode_grade(data)
    data = encode_homeOwnership(data)
    data = convert_lastCreditPullD(data)
    return data


def fill_missing_values(data):
    """
    fill missing values with mean or 0. 
    -------
    Parameter
    data: pandas dataframe.
    --------
    Returns
    data: pandas dataframe.
    """
    data['revol_util'].fillna(data['revol_util'].mean(),inplace=True)
    data['Month'].fillna(data.mode()['Month'][0],inplace=True)
    data['Year'].fillna(data.mode()['Year'][0],inplace=True)
    return data

def set_y_as_last(data,col):
    """
    Set label column to last in dataframe.
    -------
    Parameter
    data: pandas dataframe.
    co: label column.
    --------
    Returns
    data: pandas dataframe.
    """
    data['last']=data[col]
    del data[col]
    data[col]=data['last']
    del data['last']

# -------- Execution / main --------- #
def output_final(path,col='default_ind'):
    """
    Pre-process data using all the functions.
    -------
    Parameter
    data: pandas dataframe.
    col: label column.
    --------
    Returns
    data: pandas dataframe.
    """
    data = txt_to_dataframe(path)
    data = delete_col(data,800000)
    data = remove_lowCor(data,0.02,col)
    data = encode_all(data)
    data = delete_objType(data)
    data = fill_missing_values(data)
    data = delete_extraCols(data)
    set_y_as_last(data,col)
    return data
    