import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from modAL.models import ActiveLearner, Committee

def initialise_learner(X_train,y_train,n_members,model=RandomForestClassifier()):
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
    c = learner.predict(X_train)
    unqueried_score = accuracy_score(y_train,c)
    return(unqueried_score)

def query_by_committee(learner,X_pool,y_pool,X_train,y_train,unqueried_score,n_queries=200):
    perf_his = [unqueried_score]
    for index in range(n_queries):
        query_index, query_instance = learner.query(X_pool)
        X,y = X_pool[query_index].reshape(1,-1),y_pool[query_index].reshape(1,)
        learner.teach(X=X,y=y)
        X_pool,y_pool =np.delete(X_pool,query_index,axis=0),np.delete(y_pool,query_index)
        model_acc = learner.score(X_train,y_train)
        print("accuracy after "+str(index)+" Queries :"+str(model_acc))
        perf_his.append(model_acc)
    performance_history = []
    return learner, performance_history,X_pool,y_pool

def predict(committee,X_test):
    return committee.predict(X_test)

