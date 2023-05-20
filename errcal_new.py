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


  
    
allerrstat=pickle.load(open("allerrstat.pkl","rb"))

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
     
   #%%     
    
        # Computationtime (sec)=[12.05,8.48,1034.29,11.04,5.56]
        # 0.36,4.35,3.19,0.37,0.92]
sim_trn=[12.05,8.48,1034.29,11.04,5.56]
sim_tst=[0.36,4.35,3.19,0.37,0.92]
    # # sim_tstMS=[]
    # for ww in sim_tstS:
    #     sim_tstMS.append(ww)
    
stats_trn2=np.array([RMSE_trn,R2_trn,MAE_trn,bias_trn,std_trn])   
stats_tst2=np.array([RMSE_tst,R2_tst,MAE_tst,bias_tst,std_tst])  
    
    
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
    for ii in range(0,6):
          file1.write("{fstC} & {tr} & {C:3.2f} &{R:3.2f} & {L:3.2f} & {X:3.2f} \\\\".format(fstC=fstcol[ii],tr='Train',
                       C=conv_trn[ii],R=stats_trn[ii,0],L=stats_trn[ii,1],X=stats_trn[ii,2])) 
          file1.write('\n')
          file1.write(" & {tr} & {C:3.2f} &{R:3.2f} & {L:3.2f} & {X:3.2f} \\\\".format(tr='Test',
                       C=conv_tst[ii],R=stats_tst[ii,0],L=stats_tst[ii,1],X=stats_tst[ii,2])) 
        
        # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));

      # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));
      # file1.write('%s &  %3.2f & %3.2f &  %3.2f & %3.2f \\\\',algnme{ii},varmatpaperflpadd_nocbr(ii,:));
          file1.write('\n')

    