# In[1]
# Packages and Basic Information Setup
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from bayes_opt import BayesianOptimization
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
# Ignore Weak Warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Set Fonts
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12) # Song
plt.rcParams['font.family'] = 'Times New Roman' # Times New Roman
# In[2]
# Read Data
# Divide Training and Test Sets
# Set Random Seeds to Ensure Experimental Reproducibility
df = pd.read_csv('./TrainData.csv', encoding='GBK')
IN = df.iloc[:, 0:35]
OUT = df.iloc[:, -1]
Xall_train, Xall_test, Yall_train, Yall_test = train_test_split(IN, OUT, test_size=0.3, random_state=42)
# In[3]
# Setting Up Empty Arrays to Retain Calculation Process Data
xgboost_r2_scores = []
xgboost_mse_scores = []
adaboost_r2_scores = []
adaboost_mse_scores = []
catboost_r2_scores = []
catboost_mse_scores = []
gbdt_r2_scores = []
gbdt_mse_scores = []
randomforest_r2_scores = []
randomforest_mse_scores = []
bp_r2_scores = []
bp_mse_scores = []
knn_r2_scores = []
knn_mse_scores = []
# In[4]
# # Define a Plain Bayesian Optimization Function with 5-fold Cross-Validation for each of the seven models
# XGBoost
def xgboost(n_estimators, learning_rate, subsample, colsample_bytree, max_depth, gamma):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample,
                         colsample_bytree=colsample_bytree, max_depth=max_depth, gamma=gamma)
    mse = -cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(model, X_train, Y_train, cv=5).mean()
    xgboost_r2_scores.append(r2)
    xgboost_mse_scores.append(mse)
    return r2
# AdaBoost
def adaboost(n_estimators, learning_rate, random_state):
    n_estimators = int(n_estimators)
    random_state = int(random_state)
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    mse = -cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(model, X_train, Y_train, cv=5).mean()
    adaboost_r2_scores.append(r2)
    adaboost_mse_scores.append(mse)
    return r2
# CatBoost
def catboost(n_estimators, learning_rate, subsample, colsample_bylevel, max_depth, l2_leaf_reg):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    model = CatBoostRegressor(iterations=n_estimators, learning_rate=learning_rate, subsample=subsample, colsample_bylevel=colsample_bylevel,
                              depth=max_depth, l2_leaf_reg=l2_leaf_reg, verbose=False)
    mse = -cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(model, X_train, Y_train, cv=5).mean()
    catboost_r2_scores.append(r2)
    catboost_mse_scores.append(mse)
    return r2
# GBDT
def gbdt(n_estimators, learning_rate, subsample, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_leaf = int(min_samples_leaf)
    min_samples_split = int(min_samples_split)
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample,
                                      max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    mse = -cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(model, X_train, Y_train, cv=5).mean()
    gbdt_r2_scores.append(r2)
    gbdt_mse_scores.append(mse)
    return r2
# RandomForest
def random_forest(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    mse = -cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(model, X_train, Y_train, cv=5).mean()
    randomforest_r2_scores.append(r2)
    randomforest_mse_scores.append(mse)
    return r2
# BP Neural Network
def bp(hidden_layer_sizes1, hidden_layer_sizes2, alpha, learning_rate_init, momentum):
    hidden_layer_sizes1 = int(hidden_layer_sizes1)
    hidden_layer_sizes2 = int(hidden_layer_sizes2)
    model = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes1, hidden_layer_sizes2), max_iter=200, alpha = alpha,
                         learning_rate_init=learning_rate_init, momentum=momentum, activation='relu', solver='adam', learning_rate='adaptive')
    mse = -cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(model, X_train, Y_train, cv=5).mean()
    bp_r2_scores.append(r2)
    bp_mse_scores.append(mse)
    return r2
# KNN
def knn(n_neighbors):
    n_neighbors = int(n_neighbors)
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform')
    mse = -cross_val_score(model, X_train, Y_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2 = cross_val_score(model, X_train, Y_train, cv=5).mean()
    knn_r2_scores.append(r2)
    knn_mse_scores.append(mse)
    return r2
# # Define Hyperparameters for each of the seven models
# XGBoost
params_xgboost = {'n_estimators': (100, 1000),
                  'learning_rate': (0.01, 0.3),
                  'subsample': (0.5, 1),
                  'colsample_bytree': (0.5, 1),
                  'max_depth': (3, 10),
                  'gamma': (0, 1)}
# AdaBoost
params_adaboost = {'n_estimators': (50, 1000),
                   'learning_rate': (0.01, 1),
                   'random_state': (0, 100)}
# CatBoost
params_catboost = {'n_estimators': (100, 1000),
                   'learning_rate': (0.01, 0.3),
                   'subsample': (0.5, 1),
                   'colsample_bylevel': (0.5, 1),
                   'max_depth': (3, 10),
                   'l2_leaf_reg': (1, 10)}
# GBDT
params_gbdt = {'n_estimators': (100, 1000),
               'learning_rate': (0.01, 0.3),
               'subsample': (0.5, 1),
               'max_depth': (3, 10),
               'min_samples_split': (2, 20),
               'min_samples_leaf': (1, 10)}
# RandomForest
params_randomforest = {'n_estimators': (100, 1000),
                       'max_depth': (3, 10),
                       'min_samples_split': (2, 20),
                       'min_samples_leaf': (1, 10)}
# BP Nerual Network
params_bp = {'hidden_layer_sizes1': (1, 64),
             'hidden_layer_sizes2': (1, 64),
             'alpha': (0.0001, 0.01),
             'learning_rate_init' : (0.0001, 0.01),
             'momentum': (0.5, 0.9)}
# KNN
params_knn = {'n_neighbors': (1, 20),}
# In[5]
# # Classification based on Moisture Content
for i in range(0):
    print(f"The {i+1}th Round of Optimization Prediction is being performed")
    start_index = i * 1204
    end_index = (i + 1) * 1204
    input_data = df.iloc[start_index : end_index, :]
    X = input_data.iloc[:, 1:35]
    Y = input_data.iloc[:, -1]
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.3, random_state=42)
# # Bayesian Optimization
    optimizer_xgboost = BayesianOptimization(f=xgboost, pbounds=params_xgboost, random_state=42)
    optimizer_adaboost = BayesianOptimization(f=adaboost, pbounds=params_adaboost, random_state=42)
    optimizer_catboost = BayesianOptimization(f=catboost, pbounds=params_catboost, random_state=42)
    optimizer_gbdt = BayesianOptimization(f=gbdt, pbounds=params_gbdt, random_state=42)
    optimizer_randomforest = BayesianOptimization(f=random_forest, pbounds=params_randomforest, random_state=42)
    optimizer_bp = BayesianOptimization(f=bp, pbounds=params_bp, random_state=42)
    optimizer_knn = BayesianOptimization(f=knn, pbounds=params_knn, random_state=42)
# # Start Optimization with 20 Iterations
    optimizer_xgboost.maximize(init_points=10, n_iter=10)
    optimizer_adaboost.maximize(init_points=10, n_iter=10)
    optimizer_catboost.maximize(init_points=10, n_iter=10)
    optimizer_gbdt.maximize(init_points=10, n_iter=10)
    optimizer_randomforest.maximize(init_points=10, n_iter=10)
    optimizer_bp.maximize(init_points=10, n_iter=10)
    optimizer_knn.maximize(init_points=10, n_iter=10)
# # Output the Optimal Combination of Hyperparameters
    print(optimizer_xgboost.max)
    print(optimizer_adaboost.max)
    print(optimizer_catboost.max)
    print(optimizer_gbdt.max)
    print(optimizer_randomforest.max)
    print(optimizer_bp.max)
    print(optimizer_knn.max)
# # Store Optimal Hyperparameter Combinations as json files for Later Use
    best_params_xgboost = optimizer_xgboost.max['params']
    best_params_adaboost = optimizer_adaboost.max['params']
    best_params_catboost = optimizer_catboost.max['params']
    best_params_gbdt = optimizer_gbdt.max['params']
    best_params_randomforest = optimizer_randomforest.max['params']
    best_params_bp = optimizer_bp.max['params']
    best_params_knn = optimizer_knn.max['params']
    with open(os.path.join('./Parameters',f"best_params_xgboost_{i}.json"), "w") as f:
        json.dump(best_params_xgboost, f)
    with open(os.path.join('./Parameters',f"best_params_adaboost_{i}.json"), "w") as f:
        json.dump(best_params_adaboost, f)
    with open(os.path.join('./Parameters',f"best_params_catboost_{i}.json"), "w") as f:
        json.dump(best_params_catboost, f)
    with open(os.path.join('./Parameters',f"best_params_gbdt_{i}.json"), "w") as f:
        json.dump(best_params_gbdt, f)
    with open(os.path.join('./Parameters',f"best_params_randomforest_{i}.json"), "w") as f:
        json.dump(best_params_randomforest, f)
    with open(os.path.join('./Parameters',f"best_params_bp_{i}.json"), "w") as f:
        json.dump(best_params_bp, f)
    with open(os.path.join('./Parameters',f"best_params_knn_{i}.json"), "w") as f:
        json.dump(best_params_knn, f)
# # Record MSE and R2 for later use
    with open(os.path.join('./Targets',f"xgboost_r2_scores_{i}.txt"), "w") as f:
        for score in xgboost_r2_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"xgboost_mse_scores_{i}.txt"), "w") as f:
        for score in xgboost_mse_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"adaboost_r2_scores_{i}.txt"), "w") as f:
        for score in adaboost_r2_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"adaboost_mse_scores_{i}.txt"), "w") as f:
        for score in adaboost_mse_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"catboost_r2_scores_{i}.txt"), "w") as f:
        for score in catboost_r2_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"catboost_mse_scores_{i}.txt"), "w") as f:
        for score in catboost_mse_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"gbdt_r2_scores_{i}.txt"), "w") as f:
        for score in gbdt_r2_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"gbdt_mse_scores_{i}.txt"), "w") as f:
        for score in gbdt_mse_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"randomforest_r2_scores_{i}.txt"), "w") as f:
        for score in randomforest_r2_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"randomforest_mse_scores_{i}.txt"), "w") as f:
        for score in randomforest_mse_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"bp_r2_scores_{i}.txt"), "w") as f:
        for score in bp_r2_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"bp_mse_scores_{i}.txt"), "w") as f:
        for score in bp_mse_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"knn_r2_scores_{i}.txt"), "w") as f:
        for score in knn_r2_scores:
            f.write(str(score) + "\n")
    with open(os.path.join('./Targets',f"knn_mse_scores_{i}.txt"), "w") as f:
        for score in knn_mse_scores:
            f.write(str(score) + "\n")
# # Softmax Normalized Weight Calculation
# xgboost
    xgboost_mse_scores_mean = np.mean(xgboost_mse_scores)
    xgboost_mse_scores_max = max(xgboost_mse_scores)
    xgboost_mse_scores_min = min(xgboost_mse_scores)
    xgboost_mse_scores_normalized = (xgboost_mse_scores_mean - xgboost_mse_scores_min) / (xgboost_mse_scores_max - xgboost_mse_scores_min)
    xgboost_mse_scores_log = -np.log(xgboost_mse_scores_normalized)
# adaboost
    adaboost_mse_scores_mean = np.mean(adaboost_mse_scores)
    adaboost_mse_scores_max = max(adaboost_mse_scores)
    adaboost_mse_scores_min = min(adaboost_mse_scores)
    adaboost_mse_scores_normalized = (adaboost_mse_scores_mean - adaboost_mse_scores_min) / (adaboost_mse_scores_max - adaboost_mse_scores_min)
    adaboost_mse_scores_log = -np.log(adaboost_mse_scores_normalized)
# catboost
    catboost_mse_scores_mean = np.mean(catboost_mse_scores)
    catboost_mse_scores_max = max(catboost_mse_scores)
    catboost_mse_scores_min = min(catboost_mse_scores)
    catboost_mse_scores_normalized = (catboost_mse_scores_mean - catboost_mse_scores_min) / (catboost_mse_scores_max - catboost_mse_scores_min)
    catboost_mse_scores_log = -np.log(catboost_mse_scores_normalized)
# gbdt
    gbdt_mse_scores_mean = np.mean(gbdt_mse_scores)
    gbdt_mse_scores_max = max(gbdt_mse_scores)
    gbdt_mse_scores_min = min(gbdt_mse_scores)
    gbdt_mse_scores_normalized = (gbdt_mse_scores_mean - gbdt_mse_scores_min) / (gbdt_mse_scores_max - gbdt_mse_scores_min)
    gbdt_mse_scores_log = -np.log(gbdt_mse_scores_normalized)
# randomforest
    randomforest_mse_scores_mean = np.mean(randomforest_mse_scores)
    randomforest_mse_scores_max = max(randomforest_mse_scores)
    randomforest_mse_scores_min = min(randomforest_mse_scores)
    randomforest_mse_scores_normalized = (randomforest_mse_scores_mean - randomforest_mse_scores_min) / (randomforest_mse_scores_max - randomforest_mse_scores_min)
    randomforest_mse_scores_log = -np.log(randomforest_mse_scores_normalized)
# softmax normalization
    summary = np.sum(np.exp(xgboost_mse_scores_log)+np.exp(adaboost_mse_scores_log)+np.exp(catboost_mse_scores_log)+np.exp(gbdt_mse_scores_log)+np.exp(randomforest_mse_scores_log))
    softmax_xgboost = np.exp(xgboost_mse_scores_log) / summary
    softmax_adaboost = np.exp(adaboost_mse_scores_log) / summary
    softmax_catboost = np.exp(catboost_mse_scores_log) / summary
    softmax_gbdt = np.exp(gbdt_mse_scores_log) / summary
    softmax_randomforest = np.exp(randomforest_mse_scores_log) / summary
    softmax_scores = {
        'softmax_xgboost': softmax_xgboost.tolist(),
        'softmax_adaboost': softmax_adaboost.tolist(),
        'softmax_catboost': softmax_catboost.tolist(),
        'softmax_gbdt': softmax_gbdt.tolist(),
        'softmax_randomforest': softmax_randomforest.tolist(),
    }
    # Store to JSON file
    with open(os.path.join('./Targets', f"softmax_scores_{i}.json"), "w") as f:
        json.dump(softmax_scores, f)
# # Retrain the Models with Optimal Parameters
    best_xgboost_model = XGBRegressor(n_estimators=int(best_params_xgboost['n_estimators']), learning_rate=best_params_xgboost['learning_rate'],
                                      subsample=best_params_xgboost['subsample'], colsample_bytree=best_params_xgboost['colsample_bytree'],
                                      max_depth=int(best_params_xgboost['max_depth']), gamma=best_params_xgboost['gamma'])
    best_adaboost_model = AdaBoostRegressor(n_estimators=int(best_params_adaboost['n_estimators']),
                                            learning_rate=best_params_adaboost['learning_rate'],
                                            random_state=int(best_params_adaboost['random_state']))
    best_catboost_model = CatBoostRegressor(n_estimators=int(best_params_catboost['n_estimators']), learning_rate=best_params_catboost['learning_rate'],
                                            subsample=best_params_catboost['subsample'], colsample_bylevel=best_params_catboost['colsample_bylevel'],
                                            max_depth=int(best_params_catboost['max_depth']), l2_leaf_reg=best_params_catboost['l2_leaf_reg'], verbose=False)
    best_gbdt_model = GradientBoostingRegressor(n_estimators=int(best_params_gbdt['n_estimators']), learning_rate=best_params_gbdt['learning_rate'],
                                                subsample=best_params_gbdt['subsample'], max_depth=int(best_params_gbdt['max_depth']),
                                                min_samples_split=int(best_params_gbdt['min_samples_split']), min_samples_leaf=int(best_params_gbdt['min_samples_leaf']))
    best_randomforest_model = RandomForestRegressor(n_estimators=int(best_params_randomforest['n_estimators']), max_depth=int(best_params_randomforest['max_depth']),
                                                    min_samples_split=int(best_params_randomforest['min_samples_split']), min_samples_leaf=int(best_params_randomforest['min_samples_leaf']))
    best_bp_model = MLPRegressor(hidden_layer_sizes=(int(best_params_bp['hidden_layer_sizes1']), int(best_params_bp['hidden_layer_sizes2'])),
                                 max_iter=200, alpha=best_params_bp['alpha'],
                                 learning_rate_init=best_params_bp['learning_rate_init'], momentum=best_params_bp['momentum'],
                                 activation='relu', solver='adam', learning_rate='adaptive')

    best_knn_model = KNeighborsRegressor(n_neighbors=int(best_params_knn['n_neighbors']), weights='uniform')
    best_xgboost_model.fit(X_train, Y_train)
    best_adaboost_model.fit(X_train, Y_train)
    best_catboost_model.fit(X_train, Y_train)
    best_gbdt_model.fit(X_train, Y_train)
    best_randomforest_model.fit(X_train, Y_train)
    best_bp_model.fit(X_train, Y_train)
    best_knn_model.fit(X_train, Y_train)
# # Prediction using the Best Models
    predictions_xgboost = best_xgboost_model.predict(X_test)
    predictions_adaboost = best_adaboost_model.predict(X_test)
    predictions_catboost = best_catboost_model.predict(X_test)
    predictions_gbdt = best_gbdt_model.predict(X_test)
    predictions_randomforest = best_randomforest_model.predict(X_test)
    predictions_softmse = predictions_xgboost * softmax_xgboost + predictions_adaboost * softmax_adaboost + predictions_catboost * softmax_catboost + predictions_gbdt * softmax_gbdt + predictions_randomforest * softmax_randomforest
    predictions_bp = best_bp_model.predict(X_test)
    predictions_knn = best_knn_model.predict(X_test)
# # Calculate R2 and Save
    r2_xgboost_test = r2_score(Y_test, predictions_xgboost)
    r2_catboost_test = r2_score(Y_test, predictions_catboost)
    r2_adaboost_test = r2_score(Y_test, predictions_adaboost)
    r2_gbdt_test = r2_score(Y_test, predictions_gbdt)
    r2_randomforest_test = r2_score(Y_test, predictions_randomforest)
    r2_softmse_test = r2_score(Y_test, predictions_softmse)
    r2_scores_test = {
        'r2_xgboost_test': r2_xgboost_test.tolist(),
        'r2_adaboost_test': r2_adaboost_test.tolist(),
        'r2_catboost_test': r2_catboost_test.tolist(),
        'r2_gbdt_test': r2_gbdt_test.tolist(),
        'r2_randomforest_test': r2_randomforest_test.tolist(),
        'r2_softmse_test': r2_softmse_test.tolist()
    }
    # Store to JSON file
    with open(os.path.join('./Targets', f"r2_scores_test_{i}.json"), "w") as f:
        json.dump(r2_scores_test, f)
# # Create a DataFrame to Save All Predictions
    predictions_xgboost_X = best_xgboost_model.predict(X)
    predictions_adaboost_X = best_adaboost_model.predict(X)
    predictions_catboost_X = best_catboost_model.predict(X)
    predictions_gbdt_X = best_gbdt_model.predict(X)
    predictions_randomforest_X = best_randomforest_model.predict(X)
    predictions_softmse_X = predictions_xgboost_X * softmax_xgboost + predictions_adaboost_X * softmax_adaboost + predictions_catboost_X * softmax_catboost + predictions_gbdt_X * softmax_gbdt + predictions_randomforest_X * softmax_randomforest
    predictions_bp_X = best_bp_model.predict(X)
    predictions_knn_X = best_knn_model.predict(X)
    Y = pd.DataFrame(Y.values, columns=['Actual'])
    predictions_xgboost_X = pd.DataFrame(predictions_xgboost_X, columns=['XGBoost'])
    predictions_adaboost_X = pd.DataFrame(predictions_adaboost_X, columns=['Adaboost'])
    predictions_catboost_X = pd.DataFrame(predictions_catboost_X, columns=['Catboost'])
    predictions_gbdt_X = pd.DataFrame(predictions_gbdt_X, columns=['GBDT'])
    predictions_randomforest_X = pd.DataFrame(predictions_randomforest_X, columns=['RandomForest'])
    predictions_softmse_X = pd.DataFrame(predictions_softmse_X, columns=['SoftMSE'])
    predictions_bp_X = pd.DataFrame(predictions_bp_X, columns=['BP'])
    predictions_knn_X = pd.DataFrame(predictions_knn_X, columns=['KNN'])
    result = pd.concat([Y, predictions_xgboost_X, predictions_adaboost_X, predictions_catboost_X, predictions_gbdt_X, predictions_randomforest_X, predictions_softmse_X, predictions_bp_X, predictions_knn_X],  axis=1)
    result.to_excel(f'Different Moisture Content_{i}.xlsx', index=False)
# In[6]
# All Data Bayesian Optimization
print(f"The All Data of Optimization Prediction is being performed")
# Setting Up Empty Arrays to Retain Calculation Process Data
xgboost_r2_scores_alldata = []
xgboost_mse_scores_alldata = []
adaboost_r2_scores_alldata = []
adaboost_mse_scores_alldata = []
catboost_r2_scores_alldata = []
catboost_mse_scores_alldata = []
gbdt_r2_scores_alldata = []
gbdt_mse_scores_alldata = []
randomforest_r2_scores_alldata = []
randomforest_mse_scores_alldata = []
# Define a Plain Bayesian Optimization Function with 5-fold Cross-Validation for each of the seven models
def xgboost1(n_estimators, learning_rate, subsample, colsample_bytree, max_depth, gamma):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample,
                         colsample_bytree=colsample_bytree, max_depth=max_depth, gamma=gamma)
    mse_all = -cross_val_score(model, Xall_train, Yall_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2_all = cross_val_score(model, Xall_train, Yall_train, cv=5).mean()
    xgboost_r2_scores_alldata.append(r2_all)
    xgboost_mse_scores_alldata.append(mse_all)
    return r2_all
# AdaBoost
def adaboost1(n_estimators, learning_rate, random_state):
    n_estimators = int(n_estimators)
    random_state = int(random_state)
    model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    mse_all = -cross_val_score(model, Xall_train, Yall_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2_all = cross_val_score(model, Xall_train, Yall_train, cv=5).mean()
    adaboost_r2_scores_alldata.append(r2_all)
    adaboost_mse_scores_alldata.append(mse_all)
    return r2_all
# CatBoost
def catboost1(n_estimators, learning_rate, subsample, colsample_bylevel, max_depth, l2_leaf_reg):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    model = CatBoostRegressor(iterations=n_estimators, learning_rate=learning_rate, subsample=subsample, colsample_bylevel=colsample_bylevel,
                              depth=max_depth, l2_leaf_reg=l2_leaf_reg, verbose=False)
    mse_all = -cross_val_score(model, Xall_train, Yall_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2_all = cross_val_score(model, Xall_train, Yall_train, cv=5).mean()
    catboost_r2_scores_alldata.append(r2_all)
    catboost_mse_scores_alldata.append(mse_all)
    return r2_all
# GBDT
def gbdt1(n_estimators, learning_rate, subsample, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_leaf = int(min_samples_leaf)
    min_samples_split = int(min_samples_split)
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample,
                                      max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    mse_all = -cross_val_score(model, Xall_train, Yall_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2_all = cross_val_score(model, Xall_train, Yall_train, cv=5).mean()
    gbdt_r2_scores_alldata.append(r2_all)
    gbdt_mse_scores_alldata.append(mse_all)
    return r2_all
# RandomForest
def random_forest1(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_split = int(min_samples_split)
    min_samples_leaf = int(min_samples_leaf)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    mse_all = -cross_val_score(model, Xall_train, Yall_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2_all = cross_val_score(model, Xall_train, Yall_train, cv=5).mean()
    randomforest_r2_scores_alldata.append(r2_all)
    randomforest_mse_scores_alldata.append(mse_all)
    return r2_all
# BP Neural Network
def bp1(hidden_layer_sizes1, hidden_layer_sizes2, alpha, learning_rate_init, momentum):
    hidden_layer_sizes1 = int(hidden_layer_sizes1)
    hidden_layer_sizes2 = int(hidden_layer_sizes2)
    model = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes1, hidden_layer_sizes2), max_iter=200, alpha = alpha,
                         learning_rate_init=learning_rate_init, momentum=momentum, activation='relu', solver='adam', learning_rate='adaptive')
    mse_all = -cross_val_score(model, Xall_train, Yall_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2_all = cross_val_score(model, Xall_train, Yall_train, cv=5).mean()
    bp_r2_scores.append(r2_all)
    bp_mse_scores.append(mse_all)
    return r2_all
# KNN
def knn1(n_neighbors):
    n_neighbors = int(n_neighbors)
    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='uniform')
    mse_all = -cross_val_score(model, Xall_train, Yall_train, cv=5, scoring='neg_mean_squared_error').mean()
    r2_all = cross_val_score(model, Xall_train, Yall_train, cv=5).mean()
    knn_r2_scores.append(r2_all)
    knn_mse_scores.append(mse_all)
    return r2_all
# Start Optimize
optimizer_xgboost_alldata = BayesianOptimization(f=xgboost1, pbounds=params_xgboost, random_state=42)
optimizer_adaboost_alldata = BayesianOptimization(f=adaboost1, pbounds=params_adaboost, random_state=42)
optimizer_catboost_alldata = BayesianOptimization(f=catboost1, pbounds=params_catboost, random_state=42)
optimizer_gbdt_alldata = BayesianOptimization(f=gbdt1, pbounds=params_gbdt, random_state=42)
optimizer_randomforest_alldata = BayesianOptimization(f=random_forest1, pbounds=params_randomforest, random_state=42)
optimizer_bp_alldata = BayesianOptimization(f=bp1, pbounds=params_bp, random_state=42)
optimizer_knn_alldata = BayesianOptimization(f=knn1, pbounds=params_knn, random_state=42)
# Start Optimization with 20 Iterations
optimizer_xgboost_alldata.maximize(init_points=10, n_iter=10)
optimizer_adaboost_alldata.maximize(init_points=100, n_iter=100)
optimizer_catboost_alldata.maximize(init_points=10, n_iter=10)
optimizer_gbdt_alldata.maximize(init_points=10, n_iter=10)
optimizer_randomforest_alldata.maximize(init_points=10, n_iter=10)
optimizer_bp_alldata.maximize(init_points=100, n_iter=100)
optimizer_knn_alldata.maximize(init_points=10, n_iter=10)
# Output the Optimal Combination of Hyperparameters
print(optimizer_xgboost_alldata.max)
print(optimizer_adaboost_alldata.max)
print(optimizer_catboost_alldata.max)
print(optimizer_gbdt_alldata.max)
print(optimizer_randomforest_alldata.max)
print(optimizer_bp_alldata.max)
print(optimizer_knn_alldata.max)
# Store Optimal Hyperparameter Combinations as json files for Later Use
best_params_xgboost_alldata = optimizer_xgboost_alldata.max['params']
best_params_adaboost_alldata = optimizer_adaboost_alldata.max['params']
best_params_catboost_alldata = optimizer_catboost_alldata.max['params']
best_params_gbdt_alldata = optimizer_gbdt_alldata.max['params']
best_params_randomforest_alldata = optimizer_randomforest_alldata.max['params']
best_params_bp_alldata = optimizer_bp_alldata.max['params']
best_params_knn_alldata = optimizer_knn_alldata.max['params']
with open(os.path.join('./Parameters', f"best_params_xgboost_alldata.json"), "w") as f:
    json.dump(best_params_xgboost_alldata, f)
with open(os.path.join('./Parameters', f"best_params_adaboost_alldata.json"), "w") as f:
    json.dump(best_params_adaboost_alldata, f)
with open(os.path.join('./Parameters', f"best_params_catboost_alldata.json"), "w") as f:
    json.dump(best_params_catboost_alldata, f)
with open(os.path.join('./Parameters', f"best_params_gbdt_alldata.json"), "w") as f:
    json.dump(best_params_gbdt_alldata, f)
with open(os.path.join('./Parameters', f"best_params_randomforest_alldata.json"), "w") as f:
    json.dump(best_params_randomforest_alldata, f)
with open(os.path.join('./Parameters', f"best_params_bp_alldata.json"), "w") as f:
    json.dump(best_params_bp_alldata, f)
with open(os.path.join('./Parameters', f"best_params_knn_alldata.json"), "w") as f:
    json.dump(best_params_knn_alldata, f)
# 记录MSE和MAE便于后期使用
with open(os.path.join('./Targets', f"xgboost_r2_scores_alldata.txt"), "w") as f:
    for score in xgboost_r2_scores_alldata:
        f.write(str(score) + "\n")
with open(os.path.join('./Targets', f"xgboost_mse_scores_alldata.txt"), "w") as f:
    for score in xgboost_mse_scores_alldata:
        f.write(str(score) + "\n")
with open(os.path.join('./Targets', f"adaboost_r2_scores_alldata.txt"), "w") as f:
    for score in adaboost_r2_scores_alldata:
        f.write(str(score) + "\n")
with open(os.path.join('./Targets', f"adaboost_mse_scores_alldata.txt"), "w") as f:
    for score in adaboost_mse_scores_alldata:
        f.write(str(score) + "\n")
with open(os.path.join('./Targets', f"catboost_r2_scores_alldata.txt"), "w") as f:
    for score in catboost_r2_scores_alldata:
        f.write(str(score) + "\n")
with open(os.path.join('./Targets', f"catboost_mse_scores_alldata.txt"), "w") as f:
    for score in catboost_mse_scores_alldata:
        f.write(str(score) + "\n")
with open(os.path.join('./Targets', f"gbdt_r2_scores_alldata.txt"), "w") as f:
    for score in gbdt_r2_scores_alldata:
        f.write(str(score) + "\n")
with open(os.path.join('./Targets', f"gbdt_mse_scores_alldata.txt"), "w") as f:
    for score in gbdt_mse_scores_alldata:
        f.write(str(score) + "\n")
with open(os.path.join('./Targets', f"randomforest_r2_scores_alldata.txt"), "w") as f:
    for score in randomforest_r2_scores_alldata:
        f.write(str(score) + "\n")
with open(os.path.join('./Targets', f"randomforest_mse_scores_alldata.txt"), "w") as f:
    for score in randomforest_mse_scores_alldata:
        f.write(str(score) + "\n")
# Softmax Normalized Weight Calculation
# xgboost
xgboost_mse_scores_mean_alldata = np.mean(xgboost_mse_scores_alldata)
xgboost_mse_scores_max_alldata = max(xgboost_mse_scores_alldata)
xgboost_mse_scores_min_alldata = min(xgboost_mse_scores_alldata)
xgboost_mse_scores_normalized_alldata = (xgboost_mse_scores_mean_alldata - xgboost_mse_scores_min_alldata) / (xgboost_mse_scores_max_alldata - xgboost_mse_scores_min_alldata)
xgboost_mse_scores_log_alldata = -np.log(xgboost_mse_scores_normalized_alldata)
# adaboost
adaboost_mse_scores_mean_alldata = np.mean(adaboost_mse_scores_alldata)
adaboost_mse_scores_max_alldata = max(adaboost_mse_scores_alldata)
adaboost_mse_scores_min_alldata = min(adaboost_mse_scores_alldata)
adaboost_mse_scores_normalized_alldata = (adaboost_mse_scores_mean_alldata - adaboost_mse_scores_min_alldata) / (adaboost_mse_scores_max_alldata - adaboost_mse_scores_min_alldata)
adaboost_mse_scores_log_alldata = -np.log(adaboost_mse_scores_normalized_alldata)
# catboost
catboost_mse_scores_mean_alldata = np.mean(catboost_mse_scores_alldata)
catboost_mse_scores_max_alldata = max(catboost_mse_scores_alldata)
catboost_mse_scores_min_alldata = min(catboost_mse_scores_alldata)
catboost_mse_scores_normalized_alldata = (catboost_mse_scores_mean_alldata - catboost_mse_scores_min_alldata) / (catboost_mse_scores_max_alldata - xgboost_mse_scores_min_alldata)
catboost_mse_scores_log_alldata = -np.log(xgboost_mse_scores_normalized_alldata)
# gbdt
gbdt_mse_scores_mean_alldata = np.mean(gbdt_mse_scores_alldata)
gbdt_mse_scores_max_alldata = max(gbdt_mse_scores_alldata)
gbdt_mse_scores_min_alldata = min(gbdt_mse_scores_alldata)
gbdt_mse_scores_normalized_alldata = (gbdt_mse_scores_mean_alldata - gbdt_mse_scores_min_alldata) / (gbdt_mse_scores_max_alldata - gbdt_mse_scores_min_alldata)
gbdt_mse_scores_log_alldata = -np.log(gbdt_mse_scores_normalized_alldata)
# randomforest
randomforest_mse_scores_mean_alldata = np.mean(randomforest_mse_scores_alldata)
randomforest_mse_scores_max_alldata = max(randomforest_mse_scores_alldata)
randomforest_mse_scores_min_alldata = min(randomforest_mse_scores_alldata)
randomforest_mse_scores_normalized_alldata = (randomforest_mse_scores_mean_alldata - randomforest_mse_scores_min_alldata) / (randomforest_mse_scores_max_alldata - randomforest_mse_scores_min_alldata)
randomforest_mse_scores_log_alldata = -np.log(randomforest_mse_scores_normalized_alldata)
# softmax normalization
summary_alldata = np.sum(np.exp(xgboost_mse_scores_log_alldata) + np.exp(adaboost_mse_scores_log_alldata) + np.exp(catboost_mse_scores_log_alldata) + np.exp(gbdt_mse_scores_log_alldata) + np.exp(randomforest_mse_scores_log_alldata))
softmax_xgboost_alldata = np.exp(xgboost_mse_scores_log_alldata) / summary_alldata
softmax_adaboost_alldata = np.exp(adaboost_mse_scores_log_alldata) / summary_alldata
softmax_catboost_alldata = np.exp(catboost_mse_scores_log_alldata) / summary_alldata
softmax_gbdt_alldata = np.exp(gbdt_mse_scores_log_alldata) / summary_alldata
softmax_randomforest_alldata = np.exp(randomforest_mse_scores_log_alldata) / summary_alldata
softmax_scores_alldata = {
    'softmax_xgboost_alldata': softmax_xgboost_alldata.tolist(),
    'softmax_adaboost_alldata': softmax_adaboost_alldata.tolist(),
    'softmax_catboost_alldata': softmax_catboost_alldata.tolist(),
    'softmax_gbdt_alldata': softmax_gbdt_alldata.tolist(),
    'softmax_randomforest_alldata': softmax_randomforest_alldata.tolist(),
    }
# Store to JSON file
with open(os.path.join('./Targets', f"softmax_scores_alldata.json"), "w") as f:
    json.dump(softmax_scores_alldata, f)
# Retrain the Models with Optimal Parameters
best_xgboost_model_alldata = XGBRegressor(n_estimators=int(best_params_xgboost_alldata['n_estimators']),
                                      learning_rate=best_params_xgboost_alldata['learning_rate'],
                                      subsample=best_params_xgboost_alldata['subsample'],
                                      colsample_bytree=best_params_xgboost_alldata['colsample_bytree'],
                                      max_depth=int(best_params_xgboost_alldata['max_depth']),
                                      gamma=best_params_xgboost_alldata['gamma'])
best_adaboost_model_alldata = AdaBoostRegressor(n_estimators=int(best_params_adaboost_alldata['n_estimators']),
                                        learning_rate=best_params_adaboost_alldata['learning_rate'],
                                        random_state=int(best_params_adaboost_alldata['random_state']))
best_catboost_model_alldata = CatBoostRegressor(n_estimators=int(best_params_catboost_alldata['n_estimators']),
                                        learning_rate=best_params_catboost_alldata['learning_rate'],
                                        subsample=best_params_catboost_alldata['subsample'],
                                        colsample_bylevel=best_params_catboost_alldata['colsample_bylevel'],
                                        max_depth=int(best_params_catboost_alldata['max_depth']),
                                        l2_leaf_reg=best_params_catboost_alldata['l2_leaf_reg'], verbose=False)
best_gbdt_model_alldata = GradientBoostingRegressor(n_estimators=int(best_params_gbdt_alldata['n_estimators']),
                                            learning_rate=best_params_gbdt_alldata['learning_rate'],
                                            subsample=best_params_gbdt_alldata['subsample'],
                                            max_depth=int(best_params_gbdt_alldata['max_depth']),
                                            min_samples_split=int(best_params_gbdt_alldata['min_samples_split']),
                                            min_samples_leaf=int(best_params_gbdt_alldata['min_samples_leaf']))
best_randomforest_model_alldata = RandomForestRegressor(n_estimators=int(best_params_randomforest_alldata['n_estimators']),
                                                max_depth=int(best_params_randomforest_alldata['max_depth']),
                                                min_samples_split=int(best_params_randomforest_alldata['min_samples_split']),
                                                min_samples_leaf=int(best_params_randomforest_alldata['min_samples_leaf']))
best_bp_model_alldata = MLPRegressor(hidden_layer_sizes=(int(best_params_bp_alldata['hidden_layer_sizes1']), int(best_params_bp_alldata['hidden_layer_sizes2'])),
                             max_iter=200, alpha=best_params_bp_alldata['alpha'], learning_rate_init=best_params_bp_alldata['learning_rate_init'],
                             momentum=best_params_bp_alldata['momentum'], activation='relu', solver='adam', learning_rate='adaptive')
best_knn_model_alldata = KNeighborsRegressor(n_neighbors=int(best_params_knn_alldata['n_neighbors']), weights='uniform')
best_xgboost_model_alldata.fit(Xall_train, Yall_train)
best_adaboost_model_alldata.fit(Xall_train, Yall_train)
best_catboost_model_alldata.fit(Xall_train, Yall_train)
best_gbdt_model_alldata.fit(Xall_train, Yall_train)
best_randomforest_model_alldata.fit(Xall_train, Yall_train)
best_bp_model_alldata.fit(Xall_train, Yall_train)
best_knn_model_alldata.fit(Xall_train, Yall_train)
# # 同理预测所有数据并合并输入数据至新表
predictions_xgboost_all = best_xgboost_model_alldata.predict(IN)
predictions_adaboost_all = best_adaboost_model_alldata.predict(IN)
predictions_catboost_all = best_catboost_model_alldata.predict(IN)
predictions_gbdt_all = best_gbdt_model_alldata.predict(IN)
predictions_randomforest_all = best_randomforest_model_alldata.predict(IN)
predictions_bp_all = best_bp_model_alldata.predict(IN)
predictions_knn_all = best_knn_model_alldata.predict(IN)
predictions_softmse_all = predictions_xgboost_all * softmax_xgboost_alldata + predictions_adaboost_all * softmax_adaboost_alldata + predictions_catboost_all * softmax_catboost_alldata + predictions_gbdt_all * softmax_gbdt_alldata + predictions_randomforest_all * softmax_randomforest_alldata
# # 计算所有R2并保存
r2_xgboost_all = r2_score(OUT, predictions_xgboost_all)
r2_catboost_all = r2_score(OUT, predictions_catboost_all)
r2_adaboost_all = r2_score(OUT, predictions_adaboost_all)
r2_gbdt_all = r2_score(OUT, predictions_gbdt_all)
r2_randomforest_all = r2_score(OUT, predictions_randomforest_all)
r2_softmse_all = r2_score(OUT, predictions_softmse_all)
r2_scores_all = {
    'r2_xgboost_all': r2_xgboost_all.tolist(),
    'r2_adaboost_all': r2_adaboost_all.tolist(),
    'r2_catboost_all': r2_catboost_all.tolist(),
    'r2_gbdt_all': r2_gbdt_all.tolist(),
    'r2_randomforest_all': r2_randomforest_all.tolist(),
    'r2_softmse_all': r2_softmse_all.tolist()
}
# 存储到JSON文件
with open(os.path.join('./Targets', f"r2_scores_all.json"), "w") as f:
    json.dump(r2_scores_all, f)
# # 创建DataFrame保存所有值
IN_all = pd.DataFrame(IN.values, columns=['含水率', 'WOPT1', 'WOPT2', 'WOPT3', 'WOPT4', 'WOPT5', 'WOPT6', 'WOPT7', 'WOPT8',
                                          'WOPT9', 'WWPT1', 'WWPT2', 'WWPT3', 'WWPT4', 'WWPT5', 'WWPT6', 'WWPT7', 'WWPT8', 'WWPT9', 'MC1', 'MC2',
                                          'MC3', 'MC4', 'MC5', 'MC6', 'MC7', 'MC8', 'MC9', 'WWIT1', 'WWIT2', 'WWIT3', 'WWIT4', '孔隙度', '压力', '饱和度'])
OUT_all = pd.DataFrame(OUT.values, columns=['NPV'])
predictions_xgboost_all = pd.DataFrame(predictions_xgboost_all, columns=['XGBoost'])
predictions_adaboost_all = pd.DataFrame(predictions_adaboost_all, columns=['Adaboost'])
predictions_catboost_all = pd.DataFrame(predictions_catboost_all, columns=['Catboost'])
predictions_gbdt_all = pd.DataFrame(predictions_gbdt_all, columns=['GBDT'])
predictions_randomforest_all = pd.DataFrame(predictions_randomforest_all, columns=['RandomForest'])
predictions_softmse_all = pd.DataFrame(predictions_softmse_all, columns=['SoftMSE'])
predictions_bp_all = pd.DataFrame(predictions_bp_all, columns=['BP'])
predictions_knn_all = pd.DataFrame(predictions_knn_all, columns=['KNN'])
result = pd.concat([IN_all, OUT_all, predictions_xgboost_all, predictions_adaboost_all, predictions_catboost_all, predictions_gbdt_all, predictions_randomforest_all, predictions_softmse_all, predictions_bp_all, predictions_knn_all],  axis=1)
result.to_excel(f'Total Moisture Content.xlsx', index=False)
# In[7]
# Transfer Learning
origin = pd.read_csv('./TransferGD2.csv', encoding='GBK')
X_origin = origin.iloc[:, 0:38]
Y_origin = origin.iloc[:, -1]
X_train_origin, X_test_origin, Y_train_origin, Y_test_origin = train_test_split(X_origin, Y_origin, test_size=0.3, random_state=42)
new = pd.read_csv('./NewTrainData.csv', encoding='GBK')
X_new = new.iloc[:, 0:38]
Y_new = new.iloc[:, -1]
X_train_new, X_test_new, Y_train_new, Y_test_new = train_test_split(X_new, Y_new, test_size=0.3, random_state=42)
# 加载之前训练好的最佳 XGBoost 模型的参数
with open('./Parameters/best_params_xgboost_alldata.json', 'r') as f:
    best_params_xgboost_transfer = json.load(f)
with open('./Parameters/best_params_catboost_alldata.json', 'r') as f:
    best_params_catboost_transfer = json.load(f)
with open('./Parameters/best_params_adaboost_alldata.json', 'r') as f:
    best_params_adaboost_transfer = json.load(f)
with open('./Parameters/best_params_gbdt_alldata.json', 'r') as f:
    best_params_gbdt_transfer = json.load(f)
with open('./Parameters/best_params_randomforest_alldata.json', 'r') as f:
    best_params_randomforest_transfer = json.load(f)
with open('./Targets/softmax_scores_alldata.json', 'r') as f:
    softmax_scores = json.load(f)
# 根据之前的最佳参数创建 XGBoost 模型
best_xgboost_model_origin = XGBRegressor(n_estimators=int(best_params_xgboost_transfer['n_estimators']),
                                  learning_rate=best_params_xgboost_transfer['learning_rate'],
                                  subsample=best_params_xgboost_transfer['subsample'],
                                  colsample_bytree=best_params_xgboost_transfer['colsample_bytree'],
                                  max_depth=int(best_params_xgboost_transfer['max_depth']), gamma=best_params_xgboost_transfer['gamma'])
best_adaboost_model_origin = AdaBoostRegressor(n_estimators=int(best_params_adaboost_transfer['n_estimators']),
                                        learning_rate=best_params_adaboost_transfer['learning_rate'],
                                        random_state=int(best_params_adaboost_transfer['random_state']))
best_catboost_model_origin = CatBoostRegressor(n_estimators=int(best_params_catboost_transfer['n_estimators']),
                                        learning_rate=best_params_catboost_transfer['learning_rate'],
                                        subsample=best_params_catboost_transfer['subsample'],
                                        colsample_bylevel=best_params_catboost_transfer['colsample_bylevel'],
                                        max_depth=int(best_params_catboost_transfer['max_depth']),
                                        l2_leaf_reg=best_params_catboost_transfer['l2_leaf_reg'], verbose=False)
best_gbdt_model_origin = GradientBoostingRegressor(n_estimators=int(best_params_gbdt_transfer['n_estimators']),
                                            learning_rate=best_params_gbdt_transfer['learning_rate'],
                                            subsample=best_params_gbdt_transfer['subsample'],
                                            max_depth=int(best_params_gbdt_transfer['max_depth']),
                                            min_samples_split=int(best_params_gbdt_transfer['min_samples_split']),
                                            min_samples_leaf=int(best_params_gbdt_transfer['min_samples_leaf']))
best_randomforest_model_origin = RandomForestRegressor(n_estimators=int(best_params_randomforest_transfer['n_estimators']),
                                                max_depth=int(best_params_randomforest_transfer['max_depth']),
                                                min_samples_split=int(best_params_randomforest_transfer['min_samples_split']),
                                                min_samples_leaf=int(best_params_randomforest_transfer['min_samples_leaf']))
# 原始模型训练
best_xgboost_model_origin.fit(X_train_origin, Y_train_origin)
best_adaboost_model_origin.fit(X_train_origin, Y_train_origin)
best_catboost_model_origin.fit(X_train_origin, Y_train_origin)
best_gbdt_model_origin.fit(X_train_origin, Y_train_origin)
best_randomforest_model_origin.fit(X_train_origin, Y_train_origin)
# 原始模型直接预测
predictions_xgboost_origin = best_xgboost_model_origin.predict(X_new)
predictions_catboost_origin = best_catboost_model_origin.predict(X_new)
predictions_adaboost_origin = best_adaboost_model_origin.predict(X_new)
predictions_gbdt_origin = best_gbdt_model_origin.predict(X_new)
predictions_randomforest_origin = best_randomforest_model_origin.predict(X_new)
predictions_softmse_origin = predictions_xgboost_origin * softmax_scores['softmax_xgboost_alldata'] + predictions_adaboost_origin * softmax_scores['softmax_adaboost_alldata'] + predictions_catboost_origin * softmax_scores['softmax_catboost_alldata'] + predictions_gbdt_origin * softmax_scores['softmax_gbdt_alldata'] + predictions_randomforest_origin * softmax_scores['softmax_randomforest_alldata']
# 在新数据上继续训练模型
best_xgboost_model_origin.fit(X_train_new, Y_train_new)
best_adaboost_model_origin.fit(X_train_new, Y_train_new)
best_catboost_model_origin.fit(X_train_new, Y_train_new)
best_gbdt_model_origin.fit(X_train_new, Y_train_new)
best_randomforest_model_origin.fit(X_train_new, Y_train_new)
# 重新进行预测
predictions_xgboost_new = best_xgboost_model_origin.predict(X_new)
predictions_catboost_new = best_catboost_model_origin.predict(X_new)
predictions_adaboost_new = best_adaboost_model_origin.predict(X_new)
predictions_gbdt_new = best_gbdt_model_origin.predict(X_new)
predictions_randomforest_new = best_randomforest_model_origin.predict(X_new)
predictions_softmse_new = predictions_xgboost_new * softmax_scores['softmax_xgboost_alldata'] + predictions_adaboost_new * softmax_scores['softmax_adaboost_alldata'] + predictions_catboost_new * softmax_scores['softmax_catboost_alldata'] + predictions_gbdt_new * softmax_scores['softmax_gbdt_alldata'] + predictions_randomforest_new * softmax_scores['softmax_randomforest_alldata']
# 写入excel
Y_new = pd.DataFrame(Y_new.values, columns=['Actual'])
predictions_softmse_origin = pd.DataFrame(predictions_softmse_origin, columns=['Predicted_Transfer'])
predictions_softmse_new = pd.DataFrame(predictions_softmse_new, columns=['Predicted_Origin'])
result = pd.concat([Y_new, predictions_softmse_origin, predictions_softmse_new], axis=1)
result.to_excel('GD2.xlsx', index=False)