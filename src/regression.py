from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

def model_train(X_train,y_train):
    model = GradientBoostingRegressor()
    # Define hyperparameters for grid search
    param_grid = {'learning_rate': [0.1,0.75],
                'max_depth': [4,8],
                'min_samples_leaf': [3],
                "n_estimators":[10]
                }

    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best hyperparameters: ", grid_search.best_params_)

    # Use the best hyperparameters to train the model
    model = GradientBoostingRegressor(**grid_search.best_params_)
    model.fit(X_train, y_train)
    return model

def predict(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate(y_test,y_pred):
    print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
    print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))