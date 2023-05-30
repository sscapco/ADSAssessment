import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from modAL.models import ActiveLearner, Committee

def initialise_learner(X_train,y_train,n_members,model=RandomForestClassifier()):
    """
    Initialising learner by rampling n random points and building rough decision boundary.
    -------
    Parameter
    model: sci-kit lean model for Random Forest.
    X_train: train features data.
    y_train: train labels data.
    n_members: number of points to randomly sample for initial training. 
    --------
    Returns
    learner: trained model with random points.
    X_pool: X_train data - randomly sampled points.
    y_pool: y_train data - randomly sampled points.
    """
    n_labeled_examples = X_train.shape[0]
    training_indices = np.random.randint(low=0,high=n_labeled_examples+1,size=n_members)
    # generate the pool
    X_raw = X_train
    y_raw = y_train
    X_init = X_raw[training_indices]
    Y_init = y_raw[training_indices]
    X_pool = np.delete(X_raw,training_indices,axis=0)
    y_pool = np.delete(y_raw,training_indices,axis=0)
    # initializing Committee members
    learner = ActiveLearner(estimator=model,X_training=X_init, y_training=Y_init)
    return learner,X_pool,y_pool

def assembling_committee(learner_list):
    committee = Committee(learner_list=learner_list)
    return committee

def unqueried_score(X_train,y_train,learner):
    """
    Gets score with randomly selected points before active learning.
    -------
    Parameter
    learner: sci-kit lean model for random forest trained on randomly selected points.
    X_train: train features data.
    y_train: train labels data.
    --------
    Returns
    unqueried_score: accuracy validation score.
    """
    c = learner.predict(X_train)
    unqueried_score = accuracy_score(y_train,c)
    return(unqueried_score)

def query_by_committee(learner,X_pool,y_pool,X_train,y_train,unqueried_score,n_queries=200):
    """
    Using query technique and pool data for actively train the model.
    -------
    Parameter
    learner: sci-kit lean model for Random Forest and active learning.
    X_train: train features data.
    y_train: train labels data.
    X_pool: X_train data - randomly sampled points.
    y_pool: y_train data - randomly sampled points.
    unqueried_score: accuracy validation score.
    n_queries: number of points to actively query for training.
    --------
    Returns
    learner: trained model with queried points.
    performance_history: list containing validation scores for every queried points.
    X_pool: X_train data - actively sampled points.
    y_pool: y_train data - actively sampled points.
    """
    perf_his = [unqueried_score]
    for index in range(n_queries):
        # get index of queried points
        query_index, query_instance = learner.query(X_pool)
        X,y = X_pool[query_index].reshape(1,-1),y_pool[query_index].reshape(1,)
        #train new learner
        learner.teach(X=X,y=y)
        #Delete selected query from pool
        X_pool,y_pool =np.delete(X_pool,query_index,axis=0),np.delete(y_pool,query_index)
        model_acc = learner.score(X_train,y_train)
        print("accuracy after "+str(index)+" Queries :"+str(model_acc))
        perf_his.append(model_acc)
    performance_history = []
    return learner, performance_history,X_pool,y_pool

def predict(committee,X_test):
    """
    Uses ML model and test data to predict labels.
    -------
    Parameter
    committee: Active learning model trained with queries data.
    X_test: test data.
    --------
    Returns
    y_pred: array containing labels for predictions
    """
    return committee.predict(X_test)

def random_query(learner,X_pool,y_pool,X_train,y_train,unqueried_score,n_queries=200):
    """
    Using query technique and pool data for randomly train the model.
    -------
    Parameter
    learner: sci-kit lean model for Random Forest and passive learning.
    X_train: train features data.
    y_train: train labels data.
    X_pool: X_train data - randomly sampled points.
    y_pool: y_train data - randomly sampled points.
    unqueried_score: accuracy validation score.
    n_queries: number of points to randomly query for training.
    --------
    Returns
    learner: trained model with queried points.
    performance_history: list containing validation scores for every queried points.
    X_pool: X_train data - randomly sampled points.
    y_pool: y_train data - randomly sampled points.
    """
    perf_his = [unqueried_score]
    for index in range(n_queries):
        query_index = np.random.choice(range(X_pool.shape[0]), replace=False)
        #query_index, query_instance = learner.query(X_pool)
        X,y = X_pool[query_index].reshape(1,-1),y_pool[query_index].reshape(1,)
        learner.teach(X=X,y=y)
        X_pool,y_pool =np.delete(X_pool,query_index,axis=0),np.delete(y_pool,query_index)
        model_acc = learner.score(X_train,y_train)
        print("accuracy after "+str(index)+" Queries :"+str(model_acc))
        perf_his.append(model_acc)
    performance_history = []
    return learner, performance_history,X_pool,y_pool
