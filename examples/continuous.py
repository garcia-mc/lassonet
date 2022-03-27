#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:55:58 2022

@author: carlos
"""


import os
#os.chdir('/u/garciac/lassoxnet/lassonet')
os.chdir('/home/carlos/lassoxnet/lassonet')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


from nonlin_cox_gen import nonlin_cox_gen, strong_linear

import torch

z,m,betast=strong_linear()
zt=torch.from_numpy(z)


data=nonlin_cox_gen(z,m) # 999 means infinity

y=data.round(5)


X=zt

_, true_features = X.shape
# add dummy feature
#X = np.concatenate([X, torch.bernoulli(a),torch.bernoulli(a),torch.bernoulli(a)], axis=1)
#X = torch.cat([X, torch.bernoulli(a),torch.bernoulli(a),torch.bernoulli(a)], axis=1)



#y = np.stack([y, np.random.binomial(1, 0.8, *y.shape)], axis=1)
# X = np.concatenate([X1, X2], axis=1)
# y = np.stack([T, C], axis=1)

#feature_names = list(dataset.feature_names) + ["fake"] * true_features

# standardize
X = StandardScaler().fit_transform(X)

from utils import plot_path

from interfaces import LassoNetRegressor


model = LassoNetRegressor(
    hidden_dims=(10,10),
    eps_start=0.1,
    verbose=True,
)

# X_train, X_test, y_train, y_test = train_test_split(X, y)


path = model.path(X, y)

path1=path

pathplot=path1

vlosses=[pathplot[k].val_loss for k in range(len(path1))]
losses=[pathplot[k].loss for k in range(len(path1))]
selected=[np.array(pathplot[k].selected) for k in range(len(path1))]

lambdas=[pathplot[k].lambda_ for k in range(len(path1))]

import matplotlib.pyplot as plt

plt.scatter(lambdas,losses,c='blue')
plt.scatter(lambdas,vlosses,c='red')
plt.xscale('log')

plt.show()

np.argsort(losses+vlosses)
np.argsort(vlosses)
np.argsort(losses)


# each element in path corresponds to a value of lambda

for param in model.model.parameters():
    print(param)