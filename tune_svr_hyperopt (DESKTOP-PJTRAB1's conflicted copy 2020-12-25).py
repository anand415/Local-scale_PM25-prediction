# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 22:06:08 2020

@author: anand
"""
from hpsklearn import HyperoptEstimator, svr_rbf,any_preprocessing,standard_scaler
from sklearn.datasets import load_iris
from hyperopt import tpe
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
import pickle
# import metrics
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
import timeit
import pickle 
from hyperopt import hp
from sklearn.metrics import mean_squared_error

start = timeit.default_timer()

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
AMD=loadmat('pm25_regression.mat')

X=AMD["matRadd2"][:,[0,1,2,4]]
Y=AMD["matRadd2"][:,5]
# X_s = preprocessing.scale(X)
# Y_s = preprocessing.scale(Y)
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split,RepeatedKFold 

# Xs = StandardScaler().fit_transform(X)
# Ys = StandardScaler.fit_transform(X)

size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
#shuffle the data
XR,YR=[X[idx], Y[idx]]
XR1=XR[::5,[0,1,2,3]]
YR1=YR[::5]
XR2=XR[::5,[0,1,2]]
YR2=YR[::5]
# Download the data and split into training and test sets
cv_outer = KFold(n_splits=5)
scr=[]
scrbm=[]
for train_index, test_index in cv_outer.split(XR1, YR1):
    start = timeit.default_timer()
    X_train1, X_test1 = XR1[train_index], XR1[test_index]
    y_train1, y_test1 = YR1[train_index], YR1[test_index]
    X_train2, X_test2 = XR2[train_index], XR2[test_index]
    y_train2, y_test2 = YR2[train_index], YR2[test_index]


    np.random.seed(3)
    indices = np.random.permutation(len(XR))

    
    # Instantiate a HyperoptEstimator with the search space and number of evaluations
    my_epsilon = hp.loguniform('epsilon', low=np.log(1e-3), high=np.log(1))
    my_C = hp.loguniform('C', low=np.log(1e-3), high=np.log(1000))
    my_gamma = hp.loguniform('gamma', low=np.log(1e-3), high=np.log(1000))
    my_kernel="rbf"
    if __name__ == '__main__':
        estim = HyperoptEstimator(regressor=svr_rbf('my_svr',C=my_C,gamma=my_gamma,epsilon=my_epsilon),
                                        preprocessing=[standard_scaler('my_standard_scaler',with_mean=True,with_std=True)],
                                        algo=tpe.suggest, loss_fn=mean_squared_error,
                                        max_evals=500)
        
        # Search the hyperparameter space based on the data
        
        estim.fit(X_train1, y_train1)
    
    # # Show the results
    
        scr.append(estim.score(X_test1, y_test1))
    # # 1.0
    
        scrbm.append(estim.best_model())


# model = HyperoptEstimator(regressor=any_regressor('reg'), preprocessing=any_preprocessing('preprocessing'), 
#                           algo=tpe.suggest, max_evals=50, trial_timeout=30)
# # perform the search
# model.fit(X_train, y_train)
# # summarize performance
# mae = model.score(X_test, y_test)
# print("MAE: %.3f" % mae)
# # summarize the best model
# print(model.best_model())


    # print(estim.get_params())