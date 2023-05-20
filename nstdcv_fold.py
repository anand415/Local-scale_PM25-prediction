# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 03:00:36 2020

@author: anand
"""
import pickle
from scipy.io import loadmat
import numpy as np
from quick_hyperoptt_rmse import quick_hyperopt
import sys
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split,RepeatedKFold 
from lightgbm import LGBMRegressor
from tqdm import tqdm
import pandas as pd
import timeit
from xgboost import XGBRegressor

start = timeit.default_timer()
  
AMD=loadmat('FmatctB3020.mat')
X=AMD["Fmatct_lng"][:,:4]
Y=AMD["Fmatct_lng"][:,4]

size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)
#shuffle the data
XR,YR=[X[idx], Y[idx]]

cv_outer = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
y_prederr=[]
trainind=[]
testind=[]
y_actts=[]
y_acttr=[]
xgbp_fld_param=[]
kk=0
for train_index, test_index in cv_outer.split(XR, YR):
    start = timeit.default_timer()
    X_train, X_test = XR[train_index], XR[test_index]
    y_train, y_test = YR[train_index], YR[test_index]
    kk=kk+1
    # if 1<kk<=6:
	  
# 	    [xgb_params_lng,ttrials]=pickle.load(open("xgb_nest_2000_fine{}.pkl".format(kk),"rb"))
    [xgb_params_lng,ttrials] = quick_hyperopt(X_train, y_train, 'xgb', 15,diagnostic=True)
    pickle.dump([xgb_params_lng,ttrials],open("xgb_nest_2000_fine{}.pkl".format(kk),"wb"))
      
         # print(train_index[20:30])
    # xgbp = quick_hyperopt(X_train, y_train, 'xgbm', 1000)
        # xgbp_fld_param.append(xgb_params_lng)
        # # print(xgbp.keys())
        # del xgb_params_lng['eval_metric']
        # del xgb_params_lng['objective']
        # xgb = XGBRegressor(verbose=-1,random_state=24, n_jobs=-1,**xgb_params_lng)
        # mdl = xgb.fit(X_train, y_train)
        
        # y_pred_tst = mdl.predict(X_test)
        # y_pred_trn = mdl.predict(X_train)
        # y_actts.append(y_test)
        # y_acttr.append(y_test)
        # trainind.append(train_index)
        # testind.append(test_index)
        # stop = timeit.default_timer()
        # print(stop-start)

    # pickle.dump([y_pred_tst,y_pred_trn,y_actts,y_acttr,trainind,testind],open("xgb_3000_fld1_2.pkl","wb"))
#     print("Val Acc:",auc, "Best GS Acc:",gd_search.best_score_, "Best Params:",gd_search.best_params_)
    
  
#   # Training final model
  
# model = LogisticRegression(random_state=7, C=0.001, class_weight='balanced', penalty='l2').fit(X_train, y_train)
# y_pred_prob = model.predict_proba(X_test)[:,1]
# print("AUC", metrics.roc_auc_score(y_test, y_pred_prob))
# print(metrics.confusion_matrix(y_test, y_pred))
  
# view rawnested_CV.py hosted with ❤ by GitHub