#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 12:05:55 2022

@author: carlos
"""

import os
os.chdir('/u/garciac/lassoxnet/lassonet')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


from cox_gen import cox_gen

import torch

a = torch.ones(1500,1)*0.5

zt=torch.bernoulli(a)

z=zt.numpy()

data=cox_gen(z,[3.1]) # 999 means infinity

y=data.round(5)


X=zt

_, true_features = X.shape
# add dummy feature
X = np.concatenate([X, torch.bernoulli(a),torch.bernoulli(a),torch.bernoulli(a)], axis=1)
#y = np.stack([y, np.random.binomial(1, 0.8, *y.shape)], axis=1)
# X = np.concatenate([X1, X2], axis=1)
# y = np.stack([T, C], axis=1)

#feature_names = list(dataset.feature_names) + ["fake"] * true_features

# standardize
# X = StandardScaler().fit_transform(X)

from utils import plot_path

from interfaces import LassoNetRegressor


model = LassoNetRegressor(
    hidden_dims=(3,),
    eps_start=0.1,
    verbose=True,
)

X_train, X_test, y_train, y_test = train_test_split(X, y)


path = model.path(X_train, y_train)

# each element in path corresponds to a value of lambda

for param in model.model.parameters():
    print(param)
    

  
###########
    
plot_path(model, path, X_test, y_test)

plt.savefig("diabetes.png")

plt.clf()

n_features = X.shape[1]
importances = model.feature_importances_.numpy()
order = np.argsort(importances)[::-1]
importances = importances[order]
ordered_feature_names = [feature_names[i] for i in order]
color = np.array(["g"] * true_features + ["r"] * (n_features - true_features))[order]

import matplotlib.pyplot as plt

plt.subplot(211)
plt.bar(
    np.arange(n_features),
    importances,
    color=color,
)
plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
colors = {"real features": "g", "fake features": "r"}
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.ylabel("Feature importance")

_, order = np.unique(importances, return_inverse=True)

plt.subplot(212)
plt.bar(
    np.arange(n_features),
    order + 1,
    color=color,
)
plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
plt.legend(handles, labels)
plt.ylabel("Feature order")

plt.savefig("diabetes-bar.png")



