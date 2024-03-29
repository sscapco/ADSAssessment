import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import dataPreprocessing as dpp
import randomForest as rf
import neuralNet as nn
import activeLearning as aL
import train as tr
import regression as reg

# Data file path input.
path = "C:\\Users\\qsrt\OneDrive - Capco\\Documents\\ADS\\xyz\\data\\XYZCorp_LendingData.txt"
#Model Selection : for Random Forest use 'rf', for Neural Nets use 'nn', for Active Learning use 'aL', for regression use 'reg'.
model_selection = 'reg'

#Data Pre-processing and splitting.
if model_selection =='reg':
    data= dpp.output_final(path,'int_rate')
    X_train,y_train,X_test,y_test = tr.prep_reg_data(data)
else:
    data = dpp.output_final(path)
    X_train,y_train,X_test,y_test = tr.splitXY(data,random_state=42)

#Model Training and Evaluation.
if (model_selection == 'rf'):
    model_trained = tr.train_rf(X_train,y_train)
    y_pred = rf.predict(model_trained,X_test)
    tr.evaluate(y_pred,y_test)
    tr.ROC_plot(model_trained,X_test,y_test)
elif(model_selection =='nn'):
    model_trained = tr.train_nn(X_train,y_train,batch_size=200,epoch=20)
    y_pred = nn.predict(model_trained,X_test)
    tr.evaluate(y_pred.round(),y_test)
    nn.roc(y_pred,y_test)
elif(model_selection == 'aL'):
    #Comment first line for random queries and second line for Active queries.
    committee= tr.train_al(X_train,y_train,n_members=20,n_queries=120)
    #committee= tr.train_al_random(X_train,y_train,n_members=20,n_queries=120)
    y_pred = aL.predict(committee,X_test)
    tr.evaluate(y_pred,y_test)
elif(model_selection == 'reg'):
    model_trained = reg.model_train(X_train,y_train)
    y_pred = reg.predict(model_trained,X_test)
    reg.evaluate(y_test,y_pred)
