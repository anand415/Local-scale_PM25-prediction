# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:28:17 2020

@author: user
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


start = timeit.default_timer()

AMD=loadmat('pm25_regression.mat')

X=AMD["matRadd2"][:,[0,1,2,3,4]]
Y=AMD["matRadd2"][:,5]

size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
XR=X[idx]
yR=Y[idx]
n_est=1000
LR=0.3
X_train, X_test, y_train, y_test = train_test_split(XR, yR, test_size=0.2, random_state=24)



#%%

xgb= XGBRegressor(random_state=24,n_jobs=-1)


mse5000D1S, bias5000D1S, var5000D1S = bias_variance_decomp(xgb, X_train, y_train, X_test, y_test, loss='mse', num_rounds=2000, random_seed=1)

# xgb= XGBRegressor(random_state=24,n_jobs=-1,n_estimators=200)


# mse5000DS, bias5000DS, var5000DS = bias_variance_decomp(xgb, X_train, y_train, X_test, y_test, loss='mse', num_rounds=2000, random_seed=1)


xgbpm=pickle.load(open("xgbpm5ip","rb"))
xgb= XGBRegressor(random_state=24,n_jobs=-1,n_estimators=200,**xgbpm)
mse5000HS, bias5000HS, var5000HS = bias_variance_decomp(xgb, X_train, y_train, X_test, y_test, loss='mse', num_rounds=2000, random_seed=1)
#%%