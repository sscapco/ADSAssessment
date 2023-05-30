import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve,roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import dataPreprocessing as dpp
import randomForest as rf
import neuralNet as nn
import activeLearning as aL

def splitXY(data,random_state=42):
    """
    Splits features and labels randomly
    -------
    Parameter
    data: pre-processed data.
    random_state: random reed.
    --------
    Returns
    X_train: train data containing features.
    y_train: train data containing labels.
    X_test : test data containing features.
    y_test : test data containing labels.
    """
    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.25,random_state=random_state)
    return X_train,X_test,y_train,y_test

def prep_reg_data(data):
    """
    Splits features and labels randomly for regression model.
    -------
    Parameter
    data: pre-processed data.
    --------
    Returns
    X_train: train data containing features.
    y_train: train data containing labels.
    X_test : test data containing features.
    y_test : test data containing labels.
    """
    df = data.replace ([np.inf,-np.inf],np.nan)
    df = df.fillna(1e9)
    X_train,y_train,X_test,y_test = splitXY(df)
    return X_train,y_train,X_test,y_test

def evaluate(y_pred,y_test):
    """
    Takes predicted values and real values to evaluate labels.
    -------
    Parameter
    y_pred: predicted labels.
    y_test: actual labels.
    --------
    """
    print("confusion matrix:",'\n')
    print(confusion_matrix(y_test,y_pred))
    print("",'\n')
    print(classification_report(y_test,y_pred))

def ROC_plot(model,X_test,y_test):
    """
    Takes predicted values and real values to plot ROC curve.
    -------
    Parameter
    model : ML model
    y_pred: predicted labels.
    y_test: actual labels.
    --------
    """
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    #print ROC Score
    print('\n',"ROC_AUC_SCORE :",roc_auc_score(y_test, y_pred_proba))

def train_rf(X_train,y_train):
    """
    Uses ML model and train data for training.
    -------
    Parameter
    X_train: train features data.
    y_train: train labels data.
    --------
    Returns
    model: trained model.
    """
    model = rf.model()
    model_trained = rf.train(model,X_train,y_train)
    return model_trained

def train_nn(X_train,y_train,optimizer='Adam',loss='binary_crossentropy',metrics='Accuracy',monitor= 'val_accuracy',mode='max',weights=True,epoch=50,batch_size=32,validation_split=0.2):
    """
    Uses functions from neuralNet.py to ouput trained model.
    -------
    Parameter
    refer to neuralNet.py
    --------
    Returns
    model: trained model.
    """
    model = nn.model_layers(X_train)
    model = nn.model_compilation(model,optimizer=optimizer,loss=loss,metrics=metrics)
    model_trained = nn.train(model,X_train,y_train,epoch=epoch,batch_size=batch_size,validation_split=validation_split)
    return model_trained

def train_al(X_train,y_train,n_members,n_queries,model=RandomForestClassifier()):
    """
    Uses functions from activeLearning.py to ouput trained model for active queries.
    -------
    Parameter
    refer to activeLearning.py
    --------
    Returns
    model: trained model.
    """
    learner,X_pool,y_pool = aL.initialise_learner(X_train,y_train,n_members,model=model)
    unqueried_score = aL.unqueried_score(X_train,y_train,learner)
    #committee = aL.assembling_committee(learner_list)
    learner, performance_history,X_pool,y_pool = aL.query_by_committee(learner,X_pool,y_pool,X_train,y_train,unqueried_score,n_queries=n_queries)
    return learner

def train_al_random(X_train,y_train,n_members,n_queries,model=RandomForestClassifier()):
    """
    Uses functions from activeLearning.py to ouput trained model for random queries.
    -------
    Parameter
    refer to activeLearning.py
    --------
    Returns
    model: trained model.
    """
    learner,X_pool,y_pool = aL.initialise_learner(X_train,y_train,n_members,model=model)
    unqueried_score = aL.unqueried_score(X_train,y_train,learner)
    learner, performance_history,X_pool,y_pool = aL.random_query(learner,X_pool,y_pool,X_train,y_train,unqueried_score,n_queries=n_queries)
    return learner