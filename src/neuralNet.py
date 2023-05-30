from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from modAL.models import ActiveLearner, Committee
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def model_layers(X):
    """
    Defines Neural Net architechture.
    -------
    Parameter
    X: X_train matrix.
    --------
    Returns
    model: Neural Net pre-defined model.
    """
    # build the neural network model
    model = Sequential()
    model.add(Dense((X.shape[1]+1)//2, input_shape=(X.shape[1],), activation='relu')) # Add an input shape! (features,)
    model.add(Dense((X.shape[1]+1)//4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def model_compilation(model,optimizer='Adam',loss='binary_crossentropy',metrics='Accuracy'):
    """
    Keras compilation API.
    For more info visit page: https://keras.io/api/models/model_training_apis/
    -------
    Parameter
    model: Neural-network model architechture.
    optimizer: optimization algorithm.
    loss: Loss function.
    metrics: Metrics to be evaluated during model training.
    --------
    Returns
    rf: Random Forest pre-defined model.
    """
    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model

def train(model,X_train,y_train,epoch=10,batch_size=10,validation_split=0.2):
    """
    Uses ML model and train data for training.
    For more info visit page: https://keras.io/api/models/model_training_apis/
    -------
    Parameter
    model: sci-kit lean model for neural nets.
    X_train: train features data.
    y_train: train labels data.
    epoch: Number of epoch to train the model (Epoch is an iteration over the entire X and y provided).
    batch size: Number of samples per gradient update.
    validaton_split: Fraction of training data to be used as validation data.
    --------
    Returns
    model: trained model.
    """
    model.fit(X_train,
            y_train,
            epochs=epoch, # you can set this to a big number!
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True)
    return model

def predict(model,X_test):
    """
    Uses ML model and test data to predict labels.
    -------
    Parameter
    model: sci-kit lean model for neual nets.
    X_test: test data.
    --------
    Returns
    y_pred: array containing labels for predictions
    """
    y_pred=model.predict(X_test)
    return(y_pred)

def roc(y_pred,y_test):
    """
    Takes predicted values and real values to plot ROC curve.
    -------
    Parameter
    y_pred: predicted labels.
    y_test: actual labels.
    --------
    """
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test,y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

