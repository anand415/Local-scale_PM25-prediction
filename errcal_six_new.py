# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:27:34 2021

@author: anand
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:37:40 2020

@author: user
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:19:01 2020

@author: anand
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.io import savemat,loadmat
import pickle

  
    
allerrstat=pickle.load(open("allerrstat_4c_5_chk.pkl","rb"))

MAE_trn=[]
RMSE_trn=[]
R2_trn=[]
bias_trn=[]
std_trn=[]

MAE_tst=[]
RMSE_tst=[]
R2_tst=[]
bias_tst=[]
std_tst=[]

for erst in allerrstat:
    [y_pred_tst,y_pred_trn,y_act_tst,y_act_trn,trnind,tstind]=erst
    indx_trn_all=[trnind]
    indx_tst_all=[tstind]

    act=np.hstack(np.array(y_act_trn))
    pred=np.hstack(np.array(y_pred_trn))
    RMSE_trn.append(np.sqrt(mean_squared_error(act, pred)))
    MAE_trn.append((mean_absolute_error(act, pred)))
    R2_trn.append(100*(r2_score(act, pred)))    
    bias_trn.append(np.abs((np.mean(act-pred))))
    std_trn.append(np.sqrt(np.var(act-pred)))
    
    
    
    
    act=np.hstack(np.array(y_act_tst))
    pred=np.hstack(np.array(y_pred_tst))
    RMSE_tst.append(np.sqrt(mean_squared_error(act, pred)))
    MAE_tst.append((mean_absolute_error(act, pred)))
    R2_tst.append(100*(r2_score(act, pred)))    
    bias_tst.append(np.abs((np.mean(act-pred))))
    std_tst.append(np.sqrt(np.var(act-pred)))
    
stats_trn2=np.array([RMSE_trn,R2_trn,MAE_trn,bias_trn,std_trn])   
stats_tst2=np.array([RMSE_tst,R2_tst,MAE_tst,bias_tst,std_tst]) 

sim_trn=[12.05,8.48,1034.29,11.04,5.56]
sim_tst=[0.36,4.35,3.19,0.37,0.92]

stats_tstsrt=stats_tst2[:,np.argsort(-stats_tst2[0,:])]
stats_trnsrt=stats_trn2[:,np.argsort(-stats_tst2[0,:])]

# stats_trn=fnlsrt[:,list(range(3,10,2))+list(range(1,2))].T
# stats_tst=fnlsrt[:,list(range(2,10,2))+list(range(0,1))].T
# pickle.dump(np.argsort(-fnl[:,2]),open("model_seq_4C.pkl","wb")) 
[y_pred_tst,y_pred_trn,y_act_tst,y_act_trn,trnind,tstind]=allerrstat[2]
act=np.hstack(np.array(y_act_tst))
pred=np.hstack(np.array(y_pred_tst))
# stats_trn=np.concatenate((stats_trn2, np.array(sim_trn).reshape(1,-1)),axis=0)
# stats_tst=np.concatenate((stats_tst2, np.array(sim_tst).reshape(1,-1)),axis=0)
savemat('pred_act.mat',dict(pred=pred,act=act))


[y_pred_tst,y_pred_trn,y_act_tst,y_act_trn,trnind,tstind]=allerrstat[3]
act=np.hstack(np.array(y_act_tst))
pred=np.hstack(np.array(y_pred_tst))
# stats_trn=np.concatenate((stats_trn2, np.array(sim_trn).reshape(1,-1)),axis=0)
# stats_tst=np.concatenate((stats_tst2, np.array(sim_tst).reshape(1,-1)),axis=0)
savemat('pred_actref.mat',dict(pred=pred,act=act))
     
#%%

algnme=['Train','Test','Train','Test','Train','Test','Train','Test','Train','Test']

fstcol=['RMSE ($\mu$g/m$^3$)',
'$R^2$ (\%)',
'MAE ($\mu$g/m$^3$)',
# '\multirow{2}{5em}{Bias (sec)}',
'Comp. time (sec)']
# xgb,lgb,cbr,rf
with open("comp4Cchkkkk.txt",'w',encoding='utf-8') as file1:
    for ii in range(0,4):
          file1.write("{0}  & {1:3.2f} &{2:3.2f} & {3:3.2f}& {4:3.2f}& {5:3.2f} &{6:3.2f} &&{7:3.2f} & {8:3.2f} & {9:3.2f} &{10:3.2f} &{11:3.2f}&{12:3.2f} \\\\".format(fstcol[ii],
                      stats_trnsrt[ii,0],stats_trnsrt[ii,1],stats_trnsrt[ii,2], stats_trnsrt[ii,3],stats_trnsrt[ii,5],stats_trnsrt[ii,4],stats_tstsrt[ii,0],stats_tstsrt[ii,1],stats_tstsrt[ii,2], stats_tstsrt[ii,3],stats_tstsrt[ii,5],stats_tstsrt[ii,4])) 
          file1.write('\n')
          # file1.write(" & {0} & {1:3.2f} &{2:3.2f} & {3:3.2f} & {4:3.2f} &{5:3.2f}\\\\".format('Test',
                      # ))         
        # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));

      # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));
      # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));
          file1.write('\n')    