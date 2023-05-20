# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 11:39:55 2020

@author: anand
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 21:56:58 2020

@author: anand
"""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVC
from xgboost import XGBRegressor
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import numpy as np
from sklearn.model_selection import RepeatedKFold
import mat4py as m4p
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import scipy.io
import timeit
import pickle 
from sklearn.metrics import fbeta_score, make_scorer
# from multiscorer import MultiScorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from mlxtend.evaluate import bias_variance_decomp
from sklearn.preprocessing import OneHotEncoder
from quick_hyperoptt_rmse import quick_hyperopt


start = timeit.default_timer()

AMD=loadmat('pm25_regression.mat')

X=AMD["matRadd2"][:,[0,1,2,3,4]]
Y=AMD["matRadd2"][:,5]
encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(X[:,4].reshape(-1,1))
# print(onehot)
XC=np.hstack([X[:,[0,1,2,3]],onehot])
size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
XR=XC[idx]
YR=Y[idx]
n_est=1000
LR=0.3
# [xgb_params_lng,ttrials] = quick_hyperopt(XR, YR, 'xgb', 2000,diagnostic=True)
[cb_params_lng,cttrials] = quick_hyperopt(XR, YR, 'cb', 200,diagnostic=True)
