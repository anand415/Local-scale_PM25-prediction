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
import pandas as pd

start = timeit.default_timer()

AMD=loadmat('pm25_regression.mat')

X=AMD["matRadd2"][:,[0,1,2,3,4]]
Y=AMD["matRadd2"][:,5]
encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(X[:,4].reshape(-1,1))
XC=np.hstack([X[:,[0,1,2,3]],onehot])
size = len(X)
R = np.random.RandomState(3)
idx = list(range(size))
#shuffle the data
R.shuffle(idx)

XR=X[idx]
YR=Y[idx]

XRx=XC[idx]
YRx=Y[idx]




XDRl = pd.DataFrame(XR,columns=['A','B','C','D','E'])
XDRl['E']=XDRl['E'].astype('category')
YDRl = pd.DataFrame(YR)



XDRc = pd.DataFrame(XR,columns=['A','B','C','D','E'])
XDRc['E']=XDRc['E'].astype('int')
YDRc = pd.DataFrame(YR)

cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)


# tree= DecisionTreeRegressor(random_state=24) # using the random state for reproducibility
bg= BaggingRegressor(random_state=24, n_jobs=-1)
##knn= KNeighborsRegressor()
#svm= SVC(random_state=24)
GBr = GradientBoostingRegressor(random_state=24)


rf= RandomForestRegressor(random_state=24,n_estimators=100,max_features='sqrt')
# rfmae= RandomForestRegressor(random_state=24,n_estimators=100,min_samples_split=10,criterion='mae')


# hgb= HistGradientBoostingRegressor(random_state=24,loss='least_absolute_deviation', max_iter=250)

[xgbp,tt]=pickle.load(open("xgb_hot_200.pkl","rb"))
xgb= XGBRegressor(random_state=24,n_jobs=-1,n_estimators=200,**xgbp)

[lgbp,tt]=pickle.load(open("lgb_hot_1000.pkl","rb"))
lgb = LGBMRegressor(random_state=24, n_jobs=-1,n_estimators=500,**lgbp)

[cbp,tt]=pickle.load(open("cb_hot_200.pkl","rb"))
cb= CatBoostRegressor(verbose=0,random_state=24,iterations=200,cat_features=[4],thread_count=-1,**cbp)

# # now, create a list with the objects 
models= [rf,xgb,lgb,cb]
# 
#models= [neigh,tree, xgb,bg, lgb,cbr,rf]
scores=[]
y_pred=[]
trainind=[]
testind=[]
y_act=[]
y_trainpred=[]
y_trainprederr=[]
def bbias(xArray, yArray):
    error=np.mean(xArray-yArray)
    return error
def rmsse(xArray, yArray):
    error=np.sqrt(mean_squared_error(xArray,yArray))
    return error
def sstd(xArray, yArray):
    error=np.sqrt(np.var(xArray-yArray))
    return error
def maae(xArray, yArray):
    error=mean_absolute_error(xArray,yArray)
    return error
    
scorer = {'rmse': make_scorer(rmsse, greater_is_better=True),
            'r2': 'r2',
            'mae':make_scorer(maae, greater_is_better=True),
          'bias': make_scorer(bbias, greater_is_better=True),
           # 'std': make_scorer(sstd, greater_is_better=True)
           }
filenameL=[]

scoresrf=cross_validate(rf, XR, YR, cv=cv, n_jobs=-1, scoring=scorer, return_train_score=True)
scoresx=cross_validate(xgb, XRx, YR, cv=cv, n_jobs=-1, scoring=scorer, return_train_score=True)
scoresl=cross_validate(lgb, XDRl, YR, cv=cv, n_jobs=-1, scoring=scorer, return_train_score=True)
scoresc=cross_validate(cb, XDRc, YR, cv=cv, n_jobs=-1, scoring=scorer, return_train_score=True)

scores=[scoresrf,scoresx,scoresl,scoresc]
     #%%     
asx=[]
for ll in scores:
    asx.append(np.mean(np.array([vv for ff,vv in ll.items()]),1))

fnl=np.array(asx)   
# fnl[:,[4,5]]=np.sqrt(np.abs(fnl[:,[4,5]]))
# fnl[:,[6,7]]=(np.abs(fnl[:,[6,7]]))
# fnl[:,[2,3]]=np.sqrt(np.abs(fnl[:,[2,3]]))
# fnl[:,[6,7]]=(np.abs(fnl[:,[4,5]]))
           
data = {'scoreshot' : scores}
m4p.savemat('scorescomphot.mat', data)   

print(fnl)
#%%

fnlsrt=fnl[np.argsort(-fnl[:,2]),:]
stats_trn=fnlsrt[:,list(range(3,10,2))+list(range(1,2))].T
stats_tst=fnlsrt[:,list(range(2,10,2))+list(range(0,1))].T
    
    # stats_trn=np.concatenate((stats_trn2, np.array(sim_trn).reshape(1,-1)),axis=0)
    # stats_tst=np.concatenate((stats_tst2, np.array(sim_tst).reshape(1,-1)),axis=0)
     
#%%

algnme=['Train','Test','Train','Test','Train','Test','Train','Test','Train','Test']

fstcol=['\multirow{2}{5em}{RMSE (sec)}',
'\multirow{2}{5em}{R2 (\%)}',
'\multirow{2}{5em}{MAE (sec)}',
'\multirow{2}{5em}{Bias (sec)}',
'\multirow{2}{5em}{Computation time (sec)}']
# xgb,lgb,cbr,rf
with open("myfile.txt",'w',encoding='utf-8') as file1:
    for ii in range(0,5):
          file1.write("{0} & {1} & {2:3.2f} &{3:3.2f} & {4:3.2f}& {5:3.2f} \\\\".format(fstcol[ii],'Train',
                      stats_trn[ii,0],stats_trn[ii,1],stats_trn[ii,2], stats_trn[ii,3])) 
          file1.write('\n')
          file1.write(" & {0} & {1:3.2f} &{2:3.2f} & {3:3.2f} & {4:3.2f} \\\\".format('Test',
                      stats_tst[ii,0],stats_tst[ii,1],stats_tst[ii,2], stats_tst[ii,3]))         
        # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));

      # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));
      # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));
          file1.write('\n')