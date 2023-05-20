# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:25:46 2020

@author: user
"""
from quick_hyperoptt import quick_hyperopt
# from skopt.plots import plot_evaluations
import pickle
import matplotlib.pyplot as plt

[param,trials]=pickle.load(open("xgb_param_anal100.pkl","rb"))
#%%
Aparam=cttrials.vals
keys=list(Aparam)

# plot_evaluations(trials)
AR=cttrials.losses
lsses=AR()
# lsses_ct=[i for i in lsses if i <120] 
aparamct=Aparam 
# lsses[np.array(lsses)<120]
# for ss in keys:
#     aparamct[ss]=[Aparam[ss][ii] for ii,jj in enumerate(lsses) if jj <120]
    
    
    
for ss in keys:
    plt.figure()
    plt.title(ss)
    plt.plot(Aparam[ss],lsses,'.')
    
    
    
# plt.plot(ttrials.losses())  
