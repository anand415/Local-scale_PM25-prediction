from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import joblib
from xgboost import XGBRegressor
import pickle 
#from memory_profiler import profile
import bz2
import _pickle as cPickle
import timeit
from  scipy.io import savemat,loadmat
import joblib

start = timeit.default_timer()
AMD=loadmat('pm25_regression.mat')
#%%
X=AMD["matRadd2"][:,:3]
Y=AMD["matRadd2"][:,3]

size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
XR=X[idx]
YR=Y[idx]
n_est=1000
LR=0.3


xgb = XGBRegressor(random_state=24, n_jobs=-1)
# xgb.set_params(n_estimators=100)
# lltmp=xgb.get_params() 
# n_est=lltmp['n_estimators']
# LR=lltmp['learning_rate']  
# # xgb=XGBRegressor(**xxgbp)
# # xgb.get_params()
xgb.fit(XR, YR)
print(np.mean(np.abs(YR-xgb.predict(XR))))


