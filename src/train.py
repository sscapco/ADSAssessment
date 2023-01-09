import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

def train_nn(X_train,y_train,optimizer='Adam',loss='binary_crossentropy',metrics='Accuracy',monitor= 'val_accuracy',mode='max',patience=10,weights=True,epoch=10,batch_size=10,validation_split=0.2):
    model = nn.model_layers(X_train)
    model = nn.model_compilation(model,optimizer=optimizer,loss=loss,metrics=metrics)
    es    = nn.earlyStopping(monitor=monitor,mode=mode,patience=patience,weights=weights)
    model_trained = nn.train(model,X_train,y_train,es,epoch=epoch,batch_size=batch_size,validation_split=validation_split)
    return model_trained

def train_al(X_train,y_train,n_members,n_queries,model=RandomForestClassifier()):
    learner_list,X_pool,y_pool = aL.initialise_learner(X_train,y_train,n_members,model=model)
    committee = aL.assembling_committee(learner_list)
    committee, performance_history,X_pool,y_pool = aL.query_by_committee(committee,X_pool,y_pool,X_train,y_train,n_queries=n_queries)
    return committee
