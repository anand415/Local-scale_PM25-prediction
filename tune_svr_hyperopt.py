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

# Xs = StandardScaler().fit_transform(X)
# Ys = StandardScaler.fit_transform(X)

size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
#shuffle the data
XR,YR=[X[idx], Y[idx]]
# Download the data and split into training and test sets



test_size = int(0.2 * len(YR))
np.random.seed(13)
indices = np.random.permutation(len(XR))
X_train = XR[indices[:-test_size]]
y_train = YR[indices[:-test_size]]
X_test = XR[indices[-test_size:]]
y_test = YR[indices[-test_size:]]

# Instantiate a HyperoptEstimator with the search space and number of evaluations
my_epsilon = hp.loguniform('epsilon', low=np.log(1e-3), high=np.log(1))
my_C = hp.loguniform('C', low=np.log(1e-3), high=np.log(1000))
my_gamma = hp.loguniform('gamma', low=np.log(1e-3), high=np.log(1000))
my_kernel="rbf"


# model = HyperoptEstimator(regressor=any_regressor('reg'), preprocessing=any_preprocessing('preprocessing'), 
#                           algo=tpe.suggest, max_evals=50, trial_timeout=30)
# # perform the search
# model.fit(X_train, y_train)
# # summarize performance
# mae = model.score(X_test, y_test)
# print("MAE: %.3f" % mae)
# # summarize the best model
# print(model.best_model())

if __name__ == '__main__':
    estim = HyperoptEstimator(regressor=svr_rbf('my_svr',C=my_C,gamma=my_gamma,epsilon=my_epsilon),
                                    preprocessing=[standard_scaler('my_standard_scaler',with_mean=True,with_std=True)],
                                    algo=tpe.suggest, loss_fn=mean_squared_error,
                                    max_evals=50)
    
    # Search the hyperparameter space based on the data
    
    estim.fit(X_train, y_train)

# # Show the results

    print(estim.score(X_test, y_test))
# # 1.0

    print(estim.best_model())
    # print(estim.get_params())