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
    X=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.25,random_state=random_state)
    return X_train,X_test,y_train,y_test

def prep_reg_data(data):
    df = data.replace ([np.inf,-np.inf],np.nan)
    df = df.fillna(1e9)
    X_train,y_train,X_test,y_test = splitXY(df)
    return X_train,y_train,X_test,y_test

def evaluate(y_pred,y_test):
    print("confusion matrix:",'\n')
    print(confusion_matrix(y_test,y_pred))
    print("",'\n')
    print(classification_report(y_test,y_pred))

def ROC_plot(model,X_test,y_test):
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
    model = rf.model()
    model_trained = rf.train(model,X_train,y_train)
    return model_trained

def train_nn(X_train,y_train,optimizer='Adam',loss='binary_crossentropy',metrics='Accuracy',monitor= 'val_accuracy',mode='max',weights=True,epoch=50,batch_size=32,validation_split=0.2):
    model = nn.model_layers(X_train)
    model = nn.model_compilation(model,optimizer=optimizer,loss=loss,metrics=metrics)
    model_trained = nn.train(model,X_train,y_train,epoch=epoch,batch_size=batch_size,validation_split=validation_split)
    return model_trained

def train_al(X_train,y_train,n_members,n_queries,model=RandomForestClassifier()):
    learner,X_pool,y_pool = aL.initialise_learner(X_train,y_train,n_members,model=model)
    unqueried_score = aL.unqueried_score(X_train,y_train,learner)
    #committee = aL.assembling_committee(learner_list)
    learner, performance_history,X_pool,y_pool = aL.query_by_committee(learner,X_pool,y_pool,X_train,y_train,unqueried_score,n_queries=n_queries)
    return learner
