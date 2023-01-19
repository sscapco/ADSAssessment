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

path = "C:\\Users\\qsrt\OneDrive - Capco\\Documents\\ADS\\xyz\\data\\XYZCorp_LendingData.txt"

data = dpp.output_final(path)
X_train,y_train,X_test,y_test = tr.splitXY(data)

model_selection = 'rf'

if (model_selection == 'rf'):
    model_trained = tr.train_rf(X_train,y_train)
    y_pred = rf.predict(model_trained,X_test)
    tr.evaluate(y_pred,y_test)
    tr.ROC_plot(model_trained,X_test,y_test)
elif(model_selection =='nn'):
    model_trained = tr.train_nn(X_train,y_train,batch_size=200,epoch=20)
    y_pred = nn.predict(model_trained,X_test)
    tr.evaluate(y_pred.round(),y_test)
    tr.ROC_plot(model_trained,X_test,y_test)
elif(model_selection == 'aL'):
    committee= tr.train_al(X_train,y_train,n_members=20,n_queries=30)
    y_pred = aL.predict(committee,X_test)
    tr.evaluate(y_pred,y_test)
