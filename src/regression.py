from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def model_train(X_train,y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def predict(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate(y_test,y_pred):
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))