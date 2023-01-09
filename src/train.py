import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import dataPreprocessing as dpp
import randomForest as rf
import neuralNet as nn
import activeLearning as aL

def splitXY(data):
    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    return X_train,X_test,y_train,y_test


def evaluate(y_pred,y_test):
    print("confusion matrix:",'\n')
    print(confusion_matrix(y_test,y_pred))
    print("",'\n')
    print(classification_report(y_test,y_pred))

def train_rf(X_train,y_train):
    model = rf.model()
    model_trained = rf.train(model,X_train,y_train)
    return model_trained

from sklearn.ensemble import RandomForestClassifier

def model(n_estimators=10,criterion='entropy'):
    rf=RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    return rf

def train1(model,X_train,y_train):
    model.fit(X_train,y_train)
    return(model)
