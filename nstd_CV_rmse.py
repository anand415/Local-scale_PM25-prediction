import pickle
from scipy.io import loadmat
import numpy as np
from quick_hyperoptt import quick_hyperopt
import sys
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split,RepeatedKFold
from lightgbm import LGBMRegressor
from tqdm import tqdm
import pandas as pd
import timeit

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
y_act=[]
y_trainprederr=[]
lgbp_fld_param=[]
for train_index, test_index in cv_outer.split(XR, YR):
    start = timeit.default_timer()
    X_train, X_test = XR[train_index], XR[test_index]
    y_train, y_test = YR[train_index], YR[test_index]
    print(train_index[20:30])

    # lgbp = quick_hyperopt(X_train, y_train, 'lgbm', 1000)
    # lgbp_fld_param.append(lgbp)
    # # print(lgbp.keys())
    # del lgbp['metric']
    # del lgbp['objective']
    # lgb = LGBMRegressor(verbose=-1,random_state=24, n_jobs=-1,**lgbp)
    # mdl = lgb.fit(X_train, y_train)
    
    # y_pred_tst = mdl.predict(X_test)
    # y_pred_trn = mdl.predict(X_train)
    # y_prederr.append(y_pred_tst-y_test)
    # y_act.append(y_test)
    # y_trainprederr.append(y_pred_trn-y_train)
    # trainind.append(train_index)
    # testind.append(test_index)
    # stop = timeit.default_timer()
    # print(stop-start)
#     print("Val Acc:",auc, "Best GS Acc:",gd_search.best_score_, "Best Params:",gd_search.best_params_)
    
  
#   # Training final model
  
# model = LogisticRegression(random_state=7, C=0.001, class_weight='balanced', penalty='l2').fit(X_train, y_train)
# y_pred_prob = model.predict_proba(X_test)[:,1]
# print("AUC", metrics.roc_auc_score(y_test, y_pred_prob))
# print(metrics.confusion_matrix(y_test, y_pred))
  
# view rawnested_CV.py hosted with ❤ by GitHub