import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def model(n_estimators=10,criterion='entropy'):
    rf=RandomForestClassifier(n_estimators, criterion)
    return rf

def train(model,X_train,y_train):
    model.fit(X_train,y_train)
    return(model)

def predict(model,X_test):
    y_pred=model.predict(X_test)
    return(y_pred)

def EvaluateCV(model,X_train,y_train,cv=10):
    acc=cross_val_score(estimator=model,X=X_train,y=y_train,cv=cv)
    print(acc.mean())
