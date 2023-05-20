# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 01:39:49 2020

@author: user
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 03:00:36 2020

@author: anand
"""
import pickle
from scipy.io import loadmat
import numpy as np
from quick_hyperoptt_rmse import quick_hyperopt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split,RepeatedKFold 
from tqdm import tqdm
import pandas as pd
import timeit
import joblib

start = timeit.default_timer()
  
AMD=loadmat('pm25_regression.mat')
X=AMD["matRadd2"][:,[0,1,2,4]]
Y=AMD["matRadd2"][:,5]

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

cv_outer = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)


lgbp_fld_param=[]
mdnms=['rf','xgb','xgb2','lgb','cb']
itrs=[0,50,50,50]

# rfmae= RandomForestRegressor(random_state=24,n_estimators=100,min_samples_split=10,criterion='mae')


# hgb= HistGradientBoostingRegressor(random_state=24,loss='least_absolute_deviation', max_iter=250)



# rfqr = RandomForestQuantileRegressor(n_estimators=100, random_state=0, n_jobs=-1)
# models= [rf,xgb,lgb,cbr]
param_mod=pickle.load(open("innercvparam_4C_10th","rb"))
# # now, create a list with the objects 
kk=0
param_mod.insert(0,[])
allerrstat=[]
for ff,[modname,ittr] in enumerate(zip(mdnms,itrs)):
    # params=[]
    paramss=param_mod[ff]
    y_pred_trn=[]
    trainind=[]
    testind=[]
    y_actts=[]
    y_acttr=[]
    y_pred_tst=[]
    for zz,[train_index, test_index] in enumerate(cv_outer.split(XR1, YR1)):
        start = timeit.default_timer()
        X_train1, X_test1 = XR1[train_index], XR1[test_index]
        y_train1, y_test1 = YR1[train_index], YR1[test_index]
        X_train2, X_test2 = XR2[train_index], XR2[test_index]
        y_train2, y_test2 = YR2[train_index], YR2[test_index]
        res=modname
        # res = [char for char in type(model).__name__ if char.isupper()] 
        mn=''.join(res)
        if modname!='xgb2':
            if modname=='rf':
                mdl= RandomForestRegressor(random_state=24,n_estimators=300)
            elif modname=='xgb':
                params=paramss[zz]
                del params['eval_metric']
                del params['objective']
                mdl= XGBRegressor(random_state=24,n_jobs=-1,n_estimators=300, **params)
    
            elif modname=='cb':
                params=paramss[zz]
                del params['eval_metric']
                mdl= CatBoostRegressor(verbose=0,random_state=24,iterations=200, **params)
            else:
                params=paramss[zz]
                del params['metric']
                del params['objective']
                mdl = LGBMRegressor(random_state=24,n_jobs=-1,n_estimators=1000, **params)
            mdl = mdl.fit(X_train1, y_train1)
            filename=mn+str(zz)
            joblib.dump(mdl, filename)  
            y_pred_tst.append(mdl.predict(X_test1))
            y_pred_trn.append(mdl.predict(X_train1))
            y_actts.append(y_test1)
            y_acttr.append(y_train1)
            trainind.append(train_index)
            testind.append(test_index)
            
        else:
            params=paramss[zz]
            del params['eval_metric']
            del params['objective']
            mdl= XGBRegressor(random_state=24,n_jobs=-1,n_estimators=300, **params)
            mdl = mdl.fit(X_train2, y_train2)
            filename=mn+str(zz)
            joblib.dump(mdl, filename)  
            y_pred_tst.append(mdl.predict(X_test2))
            y_pred_trn.append(mdl.predict(X_train2))
            y_actts.append(y_test2)
            y_acttr.append(y_train2)
            trainind.append(train_index)
            testind.append(test_index)
        stop = timeit.default_timer()
        print(stop-start)
        errstat=[y_pred_tst,y_pred_trn,y_actts,y_acttr,trainind,testind]
    allerrstat.append(errstat)
        # filename=mn+'_errstats'
pickle.dump(allerrstat,open('allerrstat_4c_5.pkl',"wb"))
# 


