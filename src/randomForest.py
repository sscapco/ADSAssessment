import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def model(n_estimators=10,criterion='entropy'):
    """
    Defines Random fororest as model.
    For more info visit page: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    -------
    Parameter
    n_estimators: number of trees in forest.
    criterion: Decision tree splitting measure.
    --------
    Returns
    rf: Random Forest pre-defined model.
    """
    rf=RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    return rf

def train(model,X_train,y_train):
    """
    Uses ML model and train data for training.
    -------
    Parameter
    model: sci-kit lean model for random forest.
    X_train: train features data.
    y_train: train labels data.
    --------
    Returns
    model: trained model.
    """
    model.fit(X_train,y_train)
    return(model)

def predict(model,X_test):
    """
    Uses ML model and test data to predict labels.
    -------
    Parameter
    model: sci-kit lean model for random forest.
    X_test: test data/
    --------
    Returns
    y_pred: array containing labels for predictions
    """
    y_pred=model.predict(X_test)
    return(y_pred)

def EvaluateCV(model,X_train,y_train,cv=10):
    """
    Uses ML model and train data for training using cross-validations.
    -------
    Parameter
    model: sci-kit lean model for random forest.
    X_train: train features data.
    y_train: train labels data.
    cv: number of cross validations / splits.
    --------
    Returns
    acc: returns the validation accuracry score.
    """
    acc=cross_val_score(estimator=model,X=X_train,y=y_train,cv=cv)
    print(acc.mean())
