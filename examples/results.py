#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:33:45 2022

@author: garciac
"""

import matplotlib.pyplot as plt

reference=data[(z==0).squeeze(),5]

treatment=data[(z==1).squeeze(),5]

from statsmodels.distributions.empirical_distribution import ECDF
ecdf0 = ECDF(reference)
ecdf1 = ECDF(treatment)

theta=np.matrix([3.1,0,0,0]).transpose()
Xt=torch.from_numpy(X)

# in case it wasn't enough at the end of lassonet
# with torch.no_grad():
#    model.mnonpar_full.fit(model.model.forward(Xt),50)

Jinford=model.mnonpar_full.JJ

gamma=5
lam=0.8






H0=np.asarray((lam/gamma)*np.multiply(np.asmatrix(np.exp(gamma*Jinford[0,Jinford[7,:]==0])-1).transpose(),np.exp(np.matmul(X[Jinford[1,Jinford[7,:]==0].astype(int),:],theta))))

H1=np.asarray((lam/gamma)*np.multiply(np.asmatrix(np.exp(gamma*Jinford[0,Jinford[7,:]==1])-1).transpose(),np.exp(np.matmul(X[Jinford[1,Jinford[7,:]==1].astype(int),:],theta))))


plt.plot(Jinford[0,Jinford[7,:]==0], H0,c='blue')
plt.plot(Jinford[0,Jinford[7,:]==1], H1,c='red')

#plt.plot(Jinford[0,Jinford[7,:]==0], -np.log(1-ecdf0(Jinford[0,Jinford[7,:]==0])),c='green')
#plt.plot(Jinford[0,Jinford[7,:]==1], -np.log(1-ecdf1(Jinford[0,Jinford[7,:]==1])),c='orange')


q=np.multiply(np.asmatrix(Jinford[6,Jinford[7,:]==1]).transpose(),np.exp(model.model.forward(Xt[Jinford[1,Jinford[7,:]==1].astype(int),:]).detach().numpy()))
q0=np.multiply(np.asmatrix(Jinford[6,Jinford[7,:]==0]).transpose(),np.exp(model.model.forward(Xt[Jinford[1,Jinford[7,:]==0].astype(int),:]).detach().numpy()))
# NEVER FORGET WE ARE ESTIMATING H_ZERO!!!!!!

plt.scatter(Jinford[0,Jinford[7,:]==0],np.asarray(q0),s=0.5,c='blue')
plt.scatter(Jinford[0,Jinford[7,:]==1],np.asarray(q),s=0.5,c='red')



plt.show()