# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:45:26 2020

@author: anand
"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from  sklearn.preprocessing  import StandardScaler
import numpy as np
from scipy.io import loadmat
from sklearn.pipeline import Pipeline

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
AMD=loadmat('pm25_regression.mat')

X=AMD["matRadd2"][:,[0,1,2,4]]
Y=AMD["matRadd2"][:,5]
# X_s = preprocessing.scale(X)
# Y_s = preprocessing.scale(Y)

# Xs = preprocessing.StandardScaler(X).fit
# Ys = preprocessing.StandardScaler(Y).fit

size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
#shuffle the data
XR,YR=[X[idx], Y[idx]]
#%%
pipe = Pipeline([
        ('scale', StandardScaler()),
        ('reg', SVR())])# svr = SVR(kernel='rbf', C=1e1, gamma=0.1)

C = [(x) for x in np.linspace(start = 0.5, stop = 5, num = 10)]
epsilon = [(x) for x in np.linspace(start = 0.1, stop = 0.3, num = 3)]

param_grid = dict(reg__epsilon=epsilon,
                  reg__C=np.logspace(-4, 1, 6),
                  reg__kernel=['rbf','linear','poly'])

svr_random = RandomizedSearchCV(estimator = pipe, param_distributions = param_grid, n_iter = 1, cv = 2, verbose=2, random_state=42, n_jobs = -1)

svr_random.fit(XR, YR)
print(svr_random.best_score_)
print(svr_random.cv_results_)