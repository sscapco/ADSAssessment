import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from modAL.models import ActiveLearner, Committee

def initialise_learner(X_train,y_train,model=RandomForestClassifier()):
    # generate the pool
    X_pool = X_train
    y_pool = y_train
    # initializing Committee members
    n_members = 2
    learner_list = list()
    for member_idx in range(n_members):
        # initial training data
        n_initial = 2
        train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
        X_Train = X_pool[train_idx]
        y_Train = y_pool[train_idx]
        # creating a reduced copy of the data with the known instances removed
        X_pool = np.delete(X_pool, train_idx, axis=0)
        y_pool = np.delete(y_pool, train_idx)
        # initializing learner
        learner = ActiveLearner(estimator=model,
                            X_training=X_Train, y_training=y_Train
                            )
        learner_list.append(learner)
    return learner_list,X_pool,y_pool

def assembling_committee(learner_list):
    committee = Committee(learner_list=learner_list)
    return committee

def unqueried_score(X_train,y_train,committee):
    unqueried_score = committee.score(X_train,y_train)
    print(unqueried_score)

def query_by_committee(committee,X_pool,y_pool,X_train,y_train,n_queries=200):
    performance_history = []
    # query by committee
    for idx in range(n_queries):
        query_idx, query_instance = committee.query(X_pool)
        committee.teach(
            X=X_pool[query_idx].reshape(1, -1),
            y=y_pool[query_idx].reshape(1, )
        )
        performance_history.append(committee.score(X_train,y_train))
        # remove queried instance from pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
    return committee, performance_history,X_pool,y_pool

def predict(committee,X_test):
    return committee.predict(X_test)

