#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 09:44:17 2022

@author: carlos
"""
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt



import os
#os.chdir('/u/garciac/lassoxnet/lassonet')
os.chdir('/home/carlos/lassoxnet/lassonet')
#datos=genfromtxt('../nhanes/carlosacel.csv', delimiter=',',skip_header=0,names=True)[1: , :]
datos=carlosacelcsv
C=datos[1:,7].astype(int)
T=datos[1:,8].astype(int)

age=np.expand_dims(datos[1:,41],axis=1).astype(float)

np.max(age)


X2=datos[1:,range(60,80)].astype(float)
probably_fake=datos[1:,range(44,59)].astype(float)
#dataset = load_diabetes()

bmi=datos[1:,[21]].squeeze()
diabetes=datos[1:,[24]].squeeze()
cancer=datos[1:,[27]].squeeze()
stroke=datos[1:,[28]].squeeze()
race=datos[1:,[22]].squeeze()

np.unique(race)

b,c=np.unique(race,return_inverse=True)
shape = (len(race), len(b))

one_hot = np.zeros(shape)

rows = np.arange(len(race))

one_hot[rows, c] = 1

one_hot=one_hot[:,:-1].astype(int)

funbmi=lambda x : 1 if x=='"Normal"' else 0
funcan=lambda x : 1 if x=='"Yes"' else 0


bmi=np.expand_dims([funbmi(x) for x in bmi],axis=1).astype(int)
cancer=np.expand_dims([funcan(x) for x in cancer],axis=1).astype(int)
stroke=np.expand_dims([funcan(x) for x in stroke],axis=1).astype(int)
diabetes=np.expand_dims([funcan(x) for x in diabetes],axis=1).astype(int)


left=[t-1 for t in T]
right=T
delta1=np.zeros(len(right))
delta2=C
delta3=[1-c for c in C]
mid=(left+right)/2
X = np.concatenate([age,X2,probably_fake],axis=1)
y = np.stack([left,right,delta1,delta2,delta3,mid], axis=1)

#feature_names = list(dataset.feature_names) + ["fake"] * true_features

# standardize




# standardize
X = StandardScaler().fit_transform(X)



X = np.concatenate([bmi,cancer,stroke,diabetes,one_hot,X],axis=1)
Xnp=X

import torch
X=torch.from_numpy(X).float()



#from utils import plot_path

from interfaces import LassoNetRegressor


model = LassoNetRegressor(
    hidden_dims=(10,10),
    eps_start=0.1,
    verbose=True,
)

path = model.path(X, y)


path1=path

pathplot=path

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
np.argsort(losses)

lambdas[np.argsort(losses)[0].astype(int)]





plt.savefig("diabetes.png")

plt.clf()

n_features = X.shape[1]
importances = model.feature_importances_.numpy()
order = np.argsort(importances)[::-1]
importances = importances[order]
ordered_feature_names = [feature_names[i] for i in order]
color = np.array(["g"] * true_features + ["r"] * (n_features - true_features))[order]


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

