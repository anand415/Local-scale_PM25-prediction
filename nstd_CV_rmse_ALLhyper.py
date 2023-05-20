# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 03:00:36 2020

@author: anand
"""
import pickle
from scipy.io import loadmat
import numpy as np
from quick_hyperoptt_rmse import quick_hyperopt
# from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split,RepeatedKFold 
from tqdm import tqdm
import pandas as pd
import timeit

start = timeit.default_timer()
  
AMD=loadmat('allmat.mat')

#%%
X=AMD["allmat"][:,:-1]
Y=AMD["allmat"][:,7]
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


cv_outer = KFold(n_splits=5, random_state=42)

y_prederr=[]
trainind=[]
testind=[]
y_actts=[]
y_acttr=[]
lgbp_fld_param=[]
# mdnms=['xgb','xgb2','lgbm','cb']
mdnms=['xgb']

itrs=[100,100,100,50]

# rf= RandomForestRegressor(random_state=24,n_estimators=300)
# rfmae= RandomForestRegressor(random_state=24,n_estimators=100,min_samples_split=10,criterion='mae')


# hgb= HistGradientBoostingRegressor(random_state=24,loss='least_absolute_deviation', max_iter=250)


xgb= XGBRegressor(random_state=24,n_jobs=-1,n_estimators=300, objective='reg:squarederror')
xgb2= XGBRegressor(random_state=24,n_jobs=-1,n_estimators=300, objective='reg:squarederror')
# lgb = LGBMRegressor(random_state=24, n_jobs=-1,n_estimators=1000)
# cb= CatBoostRegressor(verbose=0,random_state=24)

#svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
# neigh = KNeighborsRegressor(n_neighbors=1)
# rfqr = RandomForestQuantileRegressor(n_estimators=100,
    # random_state=0, n_jobs=-1)
# models= [xgb,xgb2,lgb,cb]

# # now, create a list with the objects 
kk=0
param_mod=[]
for modname,ittr in zip(mdnms,itrs):
    params=[]
    for train_index, test_index in cv_outer.split(XR1, YR1):
        start = timeit.default_timer()
        X_train1, X_test1 = XR1[train_index], XR1[test_index]
        y_train1, y_test1 = YR1[train_index], YR1[test_index]
        X_train2, X_test2 = XR2[train_index], XR2[test_index]
        y_train2, y_test2 = YR2[train_index], YR2[test_index]
        # A=(type(model).__name__)
        # res = [char for char in type(model).__name__ if char.isupper()] 
        # mn=''.join(res)
        if modname=='xgb2':
           params.append(quick_hyperopt(X_train2, y_train2, 'xgb', ittr))
        else:
           params.append(quick_hyperopt(X_train1, y_train1, modname, ittr))
    param_mod.append(params)  
pickle.dump(param_mod,open("innercvparam_4C_5th","wb"))
    
        
    # lgb_params_lng,ttrials=pickle.load(,open("lgb_nest_6000_fine{}.pkl".format(kk),"rb"))
    #   # print(train_index[20:30])
    # lgbp_fld_param.append(lgb_params_lng)
    # # print(lgbp.keys())
    # del lgb_params_lng['metric']
    # del lgb_params_lng['objective']
    # lgb = LGBMRegressor(verbose=-1,random_state=24, n_jobs=-1,**lgb_params_lng)
    # mdl = lgb.fit(X_train, y_train)

    # y_pred_tst = mdl.predict(X_test)
    # y_pred_trn = mdl.predict(X_train)
    # y_actts.append(y_test)
    # y_acttr.append(y_test)
    # trainind.append(train_index)
    # testind.append(test_index)
    # stop = timeit.default_timer()
    # print(stop-start)

# pickle.dump([y_pred_tst,y_pred_trn,y_actts,y_acttr,trainind,testind],open("lgb_6000_allfolds.pkl","wb"))
# 

