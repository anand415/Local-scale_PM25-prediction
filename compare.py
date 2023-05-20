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
#%%
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
# X_train, X_test, y_train, y_test = train_test_split(XR, yR, test_size=0.2, random_state=24)


# tree= DecisionTreeRegressor(random_state=24) # using the random state for reproducibility
bg= BaggingRegressor(random_state=24, n_jobs=-1)
##knn= KNeighborsRegressor()
#svm= SVC(random_state=24)
GBr = GradientBoostingRegressor(random_state=24)


rf= RandomForestRegressor(random_state=24,n_estimators=300,max_features='sqrt')
# rfmae= RandomForestRegressor(random_state=24,n_estimators=100,min_samples_split=10,criterion='mae')


# hgb= HistGradientBoostingRegressor(random_state=24,loss='least_absolute_deviation', max_iter=250)

[xgbp,tt]=pickle.load(open("xgb_200.pkl","rb"))
xgb= XGBRegressor(random_state=24,n_jobs=-1,n_estimators=200,**xgbp)

[lgbp,tt]=pickle.load(open("lgb_500.pkl","rb"))
lgb = LGBMRegressor(random_state=24, n_jobs=-1,n_estimators=500,**lgbp)

[cbp,tt]=pickle.load(open("cb_200.pkl","rb"))
cb= CatBoostRegressor(verbose=0,random_state=24,iterations=200,thread_count=-1,**cbp)

#svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
# neigh = KNeighborsRegressor(n_neighbors=1)
# rfqr = RandomForestQuantileRegressor(n_estimators=100,
    # random_state=0, n_jobs=-1)

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
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=1)
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

# my_scorer = make_scorer(customLoss, greater_is_better=True)

filenameL=[]
# for model in models:
#       for ii,[train_index, test_index] in enumerate(cv.split(XR)):
#     # then predict on the test set
#             X_train, X_test = XR[train_index], XR[test_index]
#             y_train, y_test = yR[train_index], yR[test_index]
#             # X_testn=np.concatenate((X_test,Xspcl))
#             model.fit(X_train, y_train) # fit the model
#             A=(type(model).__name__)
#             res = [char for char in type(model).__name__ if char.isupper()] 
#             mn=''.join(res)
# #            filename=mn+str(ii)
# #            joblib.dump(model, filename)  
# #            filenameL.append(filename)
#             y_trainprederr.append(y_train-model.predict(X_train))
#             y_trainpred.append(model.predict(X_train))
#             y_pred.append(model.predict(X_test))
#             y_act.append(y_test)
#             trainind.append(train_index)
#             testind.append(test_index)
#             stop = timeit.default_timer()
#             print('Time: ', stop - start)
            
# scipy.io.savemat('testTR_err.mat', dict(predtrain=y_trainpred,prd=y_pred,act=y_act,tsstind=testind,trrind=trainind))

for model in models:
    # for train_index, test_index in cv.split(XR):
            # then predict on the test set
          scores.append(cross_validate(model, XR, yR, cv=cv, n_jobs=-1, scoring=scorer, return_train_score=True))
          # scores.append(cross_validate(model, XR, yR, cv=cv, n_jobs=-1, scoring=['neg_mean_squared_error'], return_train_score=True))

          stop = timeit.default_timer()

          print('Time: ', stop - start) 
     #%%     
asx=[]
for ll in scores:
    asx.append(np.mean(np.array([vv for ff,vv in ll.items()]),1))

fnl=np.array(asx)   
# fnl[:,[4,5]]=np.sqrt(np.abs(fnl[:,[4,5]]))
# fnl[:,[6,7]]=(np.abs(fnl[:,[6,7]]))
# fnl[:,[2,3]]=np.sqrt(np.abs(fnl[:,[2,3]]))
# fnl[:,[6,7]]=(np.abs(fnl[:,[4,5]]))
           
data = {'scores' : scores}
m4p.savemat('scorescomp.mat', data)   

print(fnl)


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
with open("myfileNohot.txt",'w',encoding='utf-8') as file1:
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