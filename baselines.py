import random

import numpy as np
from collections import Counter

from sklearn.metrics import f1_score, make_scorer, mean_squared_error
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from joblib import dump, load

def get_model(model, eval_f, ps, feature_size = None):
    if model == 'dummy':
        clf = DummyClassifier(strategy = 'stratified', random_state= 123)         
        
    elif model == 'linear':
        estimator_lr = LogisticRegression(random_state= 123, max_iter = 10000)

        #define GS hyperparameters
        param_grid_lr ={
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__fit_intercept': (True, False),
        'clf__class_weight': [None, "balanced"]
        }        
    
        #define the pipeline
        pipe_lr = Pipeline( steps = [
            ('clf', estimator_lr)
        ])

        #create the grid search instance
        clf = GridSearchCV(estimator= pipe_lr,
                            cv = ps,
                            param_grid= param_grid_lr,
                            scoring= eval_f,
                            n_jobs=-1,
                            refit = False)
    
    
    elif model == "decisiontree":
        #define the estimator
        estimator_dt = DecisionTreeClassifier(random_state= 123)
        
        
        param_grid_dt ={
            'clf__criterion': ["gini", "entropy"],
            'clf__max_depth': [None, 3, 5, 10],
            'clf__class_weight': [None, "balanced"]
        }
    
        #define the pipeline
        pipe_dt = Pipeline( steps = [
            ('clf', estimator_dt)
        ])

        #create the grid search instance
        clf = GridSearchCV(estimator= pipe_dt,
                            cv = ps,
                            param_grid= param_grid_dt,
                            scoring= eval_f,
                            n_jobs=-1,
                            refit = False)
            
    
    elif model == 'randomforest':
        #define the classifier
        estimator_rf = RandomForestClassifier(random_state= 123)

        param_grid_rf ={
            'clf__criterion': ["gini", "entropy"],
            'clf__max_depth': [None, 3, 5, 10],
            'clf__class_weight': [None, "balanced"],
            'clf__n_estimators': [16, 32, 64, 128]
        }
    
        #define the pipeline
        pipe_rf = Pipeline( steps = [
            ('clf', estimator_rf)
        ])

        #create the grid search instance
        clf = GridSearchCV(estimator= pipe_rf,
                            cv = ps,
                            param_grid= param_grid_rf,
                            scoring= eval_f,
                            n_jobs=-1,
                            refit = False)

    
    elif model == "knn":
        #define the classifier
        estimator_knn = KNeighborsClassifier()
        
        param_grid_knn = {
            'clf__n_neighbors': [3, 5, 7, 10],
            'clf__weights': ['uniform', 'distance'] 
        }
        
                #define the pipeline
        pipe_knn = Pipeline( steps = [
            ('clf', estimator_knn)
        ])

        #create the grid search instance
        clf = GridSearchCV(estimator= pipe_knn,
                            cv = ps,
                            param_grid= param_grid_knn,
                            scoring= eval_f,
                            n_jobs=-1,
                            refit = False)
    
    elif model == 'mlp':
        #mlp classifier
        estimator_mlp = MLPClassifier(max_iter=10000, 
            verbose = False, 
            random_state = 123)
        
        #define the pipeline
        pipe_mlp = Pipeline( steps = [
            ('clf', estimator_mlp)
        ])
        
        param_grid_mlp = {
            'clf__hidden_layer_sizes': [ (feature_size,), (feature_size, feature_size // 2), (feature_size // 2,)],
            'clf__activation': ['tanh', 'relu'],
            'clf__solver': ['adam'],
            'clf__learning_rate': ['adaptive'],
            'clf__learning_rate_init': [0.01, 0.001, 0.0001],
            'clf__alpha': [0.01, 0.001, 0.0001]
        }
        
        #create the grid search instance
        clf = GridSearchCV(estimator= pipe_mlp,
                            cv = ps,
                            param_grid= param_grid_mlp,
                            scoring= eval_f,
                            n_jobs=-1,
                            refit = False)
    
    return clf

def extract_gridsearch(model, gs):
    if model == 'linear':
        clf = LogisticRegression(random_state=123, 
            C=gs.best_params_['clf__C'], 
            fit_intercept=gs.best_params_['clf__fit_intercept'],
            class_weight=gs.best_params_['clf__class_weight']
        )
        
    elif model == 'decisiontree':
        clf = DecisionTreeClassifier(random_state= 123, 
                criterion= gs.best_params_['clf__criterion'],
                max_depth=gs.best_params_['clf__max_depth'],
                class_weight=gs.best_params_['clf__class_weight']
        )
    
    elif model == 'randomforest':
        clf = RandomForestClassifier(random_state=123, 
            criterion=gs.best_params_['clf__criterion'], 
            max_depth=gs.best_params_['clf__max_depth'],
            class_weight=gs.best_params_['clf__class_weight'],
            n_estimators = gs.best_params_['clf__n_estimators']
        )
    
    elif model == 'knn':
        clf = KNeighborsClassifier(
            n_neighbors=gs.best_params_['clf__n_neighbors'],
            weights=gs.best_params_['clf__weights']
        )
    
    elif model == 'mlp':
        clf = MLPClassifier(
            hidden_layer_sizes = gs.best_params_['clf__hidden_layer_sizes'],
            activation=gs.best_params_['clf__activation'],
            solver=gs.best_params_['clf__solver'],
            learning_rate=gs.best_params_['clf__learning_rate'],
            learning_rate_init=gs.best_params_['clf__learning_rate_init'],
            alpha = gs.best_params_['clf__alpha'],
            max_iter=10000, 
            verbose = False, 
            random_state = 123,
            )

    return clf

def exec_baseline(config, model, df, Xf, yf, train_id, val_id, test_id, iter):
    #generate the dataset
    train_df = df[df["id_owner"].isin(train_id)]
    val_df = df[df["id_owner"].isin(val_id)]
    test_df = df[df["id_owner"].isin(test_id)]
    X_train = train_df[Xf]
    y_train = train_df[yf]
    X_val = val_df[Xf]
    y_val = val_df[yf]
    X_test = test_df[Xf]
    y_test = test_df[yf]

    if model == 'dummy' and config['verbose'] > 0:
        print(f"\n\nInfo Distribution -- \tTrain{Counter(y_train)}\tVal{Counter(y_val)}\tTest{Counter(y_test)}\n\n")

    #scale the data
    scl = MinMaxScaler().fit(X_train)
    X_train = scl.transform(X_train)
    X_val = scl.transform(X_val)
    X_test = scl.transform(X_test)

    y_train = train_df[yf].tolist()
    y_val = val_df[yf].tolist()
    y_test = test_df[yf].tolist()

    #prepare the data for the grid-search    
    if model != 'dummy':
        #concatenate train and val for the grid-search 
        X_train_val = np.concatenate([X_train, X_val], axis = 0)
        Y_train_val = y_train + y_val

        split_index = [-1] * len(X_train) + [0] * len(X_val)
        ps = PredefinedSplit(test_fold= split_index) #this avoids random splits
    else:
        ps = None

    #trainining phase. define classifier and evaluation metric
    eval_f = make_scorer(f1_score, average = 'weighted')
    clf = get_model(model, eval_f, ps, feature_size=len(Xf))

    if model == 'dummy': #dummy is a special case and it does not require any additional operation
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train_val, Y_train_val) #fit and find best hyperparameters
        clf = extract_gridsearch(model, clf) #extract best hyperparams
        clf.fit(X_train, y_train) #fit on the best hyperparams
        
        #save the best model
        dump(clf, f'./models/{yf}_{model}_{iter}.joblib')

    #compute the aggregate score
    y_true_agg, y_pred_agg= [], []
    for u_ in test_id:
        #filter by the given user
        test_df_user = test_df[test_df["id_owner"] == u_]
        X_test_user = test_df_user[Xf]
        y_test_user = test_df_user[yf]
        
        #prepare the data
        X_test_user = scl.transform(X_test_user)

        #process y
        y_test_user = test_df_user[yf].tolist()        
        
        y_agg_proba = clf.predict_proba(X_test_user)
        y_agg_proba = np.mean(y_agg_proba, axis = 0)
        y_agg_proba = np.argmax(y_agg_proba)
        
        y_true_agg.append(y_test_user[0])
        y_pred_agg.append(y_agg_proba)

    f1_test = f1_score(y_true = y_true_agg, y_pred = y_pred_agg, average = 'weighted')

    #compute the aggregate score
    y_true_agg, y_pred_agg= [], []
    for u_ in val_id:
        #filter by the given user
        val_df_user = val_df[val_df["id_owner"] == u_]
        X_val_user = val_df_user[Xf]
        y_val_user = val_df_user[yf]
        
        #prepare the data
        X_val_user = scl.transform(X_val_user)

        #process y
        y_val_user = val_df_user[yf].tolist()        
        
        y_agg_proba = clf.predict_proba(X_val_user)
        y_agg_proba = np.mean(y_agg_proba, axis = 0)
        y_agg_proba = np.argmax(y_agg_proba)
        
        y_true_agg.append(y_val_user[0])
        y_pred_agg.append(y_agg_proba)

    f1_val = f1_score(y_true = y_true_agg, y_pred = y_pred_agg, average = 'weighted')

    #compute the aggregate score
    y_true_agg, y_pred_agg= [], []
    for u_ in train_id:
        #filter by the given user
        train_df_user = train_df[train_df["id_owner"] == u_]
        X_train_user = train_df_user[Xf]
        y_train_user = train_df_user[yf]
        
        #prepare the data
        X_train_user = scl.transform(X_train_user)

        #process y
        y_train_user = train_df_user[yf].tolist()        
        
        y_agg_proba = clf.predict_proba(X_train_user)
        y_agg_proba = np.mean(y_agg_proba, axis = 0)
        y_agg_proba = np.argmax(y_agg_proba)
        
        y_true_agg.append(y_train_user[0])
        y_pred_agg.append(y_agg_proba)

    
    f1_train = f1_score(y_true = y_true_agg, y_pred = y_pred_agg, average = 'weighted')

    return [f1_train, f1_val, f1_test]
