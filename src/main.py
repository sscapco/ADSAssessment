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
