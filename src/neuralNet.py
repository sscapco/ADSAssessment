from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
# for modeling
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from modAL.models import ActiveLearner, Committee

def model_layers(X):
    # build the neural network model
    model = Sequential()
    model.add(Dense(16, input_shape=(X.shape[1],), activation='relu')) # Add an input shape! (features,)
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def model_compilation(model,optimizer='Adam',loss='binary_crossentropy',metrics='Accuracy'):
    # compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    return model

def earlyStopping(monitor= 'val_accuracy',mode='max',patience=10,weights=True):
    # early stopping callback  
    es = EarlyStopping(monitor=monitor, mode=mode, patience=patience, restore_best_weights=weights)
    return es

def train(model,X_train,y_train,callback,epoch=20,batch_size=10,validation_split=0.2):
    model.fit(X_train,
            y_train,
            callbacks=[callback],
            epochs=epoch, # you can set this to a big number!
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=1)
    return model

def predict(model,X_test):
    y_pred=model.predict(X_test)
    return(y_pred)
