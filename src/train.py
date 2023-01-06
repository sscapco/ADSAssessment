import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import dataPreprocessing as dpp

def splitXY(data):
    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    return X_train,X_test,y_train,y_test



#data = dpp.output_final()
#X_train,X_test,y_train,y_test = splitXY(data)
#rf = model()
#y_pred = train(rf)
#evaluation(y_test,y_pred)

