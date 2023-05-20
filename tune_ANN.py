# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 16:57:40 2020

@author: anand
"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer, Categorical 
from skopt.utils import use_named_args
from skopt import gp_minimize
import pickle
from scipy.io import loadmat
import numpy as np
from quick_hyperoptt_rmse import quick_hyperopt
import sys
import warnings

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
n_features = XR.shape[1]

reg = MLPRegressor(random_state=0,)

space=[
    Categorical(['tanh','relu'],name='activation'),
    Integer(1,4,name='n_hidden_layer'),
    Integer(50,300,name='n_neurons_per_layer'),
    ]

@use_named_args(space)

def objective(**params):
    n_neurons=params['n_neurons_per_layer']
    n_layers=params['n_hidden_layer']

    # create the hidden layers as a tuple with length n_layers and n_neurons per layer
    params['hidden_layer_sizes']=(n_neurons,)*n_layers

    # the parameters are deleted to avoid an error from the MLPRegressor
    params.pop('n_neurons_per_layer')
    params.pop('n_hidden_layer')

    reg.set_params(**params)

    return -np.mean(cross_val_score(reg, XR, YR, cv=5, n_jobs=-1,
                                    scoring="neg_mean_absolute_error"))

res_gp = gp_minimize(objective, space, n_calls=10, random_state=42,n_jobs=-1)