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
import pickle
# import metrics
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
import timeit
import pickle 

start = timeit.default_timer()

n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
AMD=loadmat('pm25_regression.mat')

X=AMD["matRaddB2"][:,[0,1,2,4]]
Y=AMD["matRaddB2"][:,5]
# X_s = preprocessing.scale(X)
# Y_s = preprocessing.scale(Y)

Xs = StandardScaler().fit_transform(X)
# Ys = StandardScaler.fit_transform(X)

size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
#shuffle the data
XR,YR=[X[idx], Y[idx]]
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)

#%%
scores=[]
pipe = Pipeline([
        ('scale', StandardScaler()),
        ('reg', SVR(kernel='rbf', C=750, gamma=76,epsilon=0.01))])# svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
svr=SVR()
# kernel='rbf', C=1e4, gamma=1e2
# # for model in models:
#     # for train_index, test_index in cv.split(XR):
#             # then predict on the test set
scores.append(cross_validate(pipe, XR, YR, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error'))
# # scores.append(cross_validate(model, XR, yR, cv=cv, n_jobs=-1, scoring=['neg_mean_squared_error'], return_train_score=True))

stop = timeit.default_timer()

print('Time: ', stop - start) 
      #%%     
asx=[]
for ll in scores:
    asx.append(np.mean(np.array([vv for ff,vv in ll.items()]),1))

fnl=np.array(asx)  
# fnl[:,[2,3]]=np.sqrt(-(asx[0][2]))
# fnl[:,[4,5]]=np.sqrt(np.abs(fnl[:,[4,5]]))
# fnl[:,[6,7]]=(np.abs(fnl[:,[6,7]]))
fnl[:,[2]]=np.sqrt(np.abs(fnl[:,[2]]))
# fnl[:,[6,7]]=(np.abs(fnl[:,[4,5]]))
            # 
print(fnl)
#%%

# C = [(x) for x in np.linspace(start = 0.5, stop = 5, num = 10)]
# epsilon = [(x) for x in np.linspace(start = 0.1, stop = 0.3, num = 3)]

# param_grid = dict(reg__gamma=np.linspace(start = 135, stop = 155, num = 20),
#                   reg__C=np.linspace(start = 1120, stop = 1240, num = 20),
#                   reg__kernel=['rbf'])

# svr_random = RandomizedSearchCV(estimator = pipe, n_iter = 500, scoring='neg_mean_squared_error',param_distributions = param_grid, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# svr_random.fit(XR, YR)
# pickle.dump(svr_random,open("svrparam.pkl","wb"))
# print(svr_random.best_score_)
# print(svr_random.cv_results_)