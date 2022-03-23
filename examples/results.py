#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:33:45 2022

@author: garciac
"""

model=path[0]

JJtr=model.mnonpar_train.JJ

JJva=model.mnonpar_val.JJ

JJtr[10,:]

Jinfor

gamma=5
lam=0.8


reference=data[(z==0).squeeze(),5]

treatment=data[(z==1).squeeze(),5]

from statsmodels.distributions.empirical_distribution import ECDF
ecdf0 = ECDF(reference)
ecdf1 = ECDF(treatment)


H0=(lam/gamma)*(np.exp(gamma*Jinford[0,Jinford[7,:]==0])-1)*np.exp(np.matmul(z[Jinford[1,Jinford[7,:]==0].astype(int),:],theta))

H1=(lam/gamma)*(np.exp(gamma*Jinford[0,Jinford[7,:]==1])-1)*np.exp(np.matmul(z[Jinford[1,Jinford[7,:]==1].astype(int),:],theta))

plt.plot(Jinford[0,Jinford[7,:]==0], H0,c='blue')
plt.plot(Jinford[0,Jinford[7,:]==1], H1,c='red')

plt.plot(Jinford[0,Jinford[7,:]==0], -np.log(1-ecdf0(Jinford[0,Jinford[7,:]==0])),c='green')
plt.plot(Jinford[0,Jinford[7,:]==1], -np.log(1-ecdf1(Jinford[0,Jinford[7,:]==1])),c='orange')

plt.scatter(Jinford[0,Jinford[7,:]==0],Jinford[6,Jinford[7,:]==0],s=0.5,c='blue')
plt.scatter(Jinford[0,Jinford[7,:]==1],Jinford[6,Jinford[7,:]==1]*np.exp(np.matmul(z[Jinford[1,Jinford[7,:]==1].astype(int),:],theta)),s=0.5,c='red')

# NEVER FORGET WE ARE ESTIMATING H_ZERO!!!!!!

plt.show()