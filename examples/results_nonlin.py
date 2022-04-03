#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 14:08:25 2022

@author: carlos
"""
import matplotlib.pyplot as plt

reference=y[:,5]


from statsmodels.distributions.empirical_distribution import ECDF
ecdf0 = ECDF(reference)



Jinford=model.mnonpar_full.JJ

gamma=5
lam=0.8






H0=np.asarray((lam/gamma)*np.asmatrix(np.exp(gamma*Jinford[0,:])-1).transpose())
#H0=np.asarray((lam/gamma)*np.multiply(np.asmatrix(np.exp(gamma*Jinford[0,:])-1).transpose(),np.exp(model.model.forward(torch.zeros(len(H0),20)).detach().numpy())))
predi=torch.zeros(len(Jinford[6,:]),X.shape[1])
predi0=torch.zeros(len(Jinford[6,:]),X.shape[1])
predi[:,0]=torch.ones(len(predi))
q0=np.multiply(np.asmatrix(Jinford[6,:]).transpose(),np.exp(model.model.forward(predi0).detach().numpy()))
q1=np.multiply(np.asmatrix(Jinford[6,:]).transpose(),np.exp(model.model.forward(predi).detach().numpy()))


#plt.plot(Jinford[0,:], H0,c='grey')
#plt.scatter(Jinford[0,:],np.asarray(q0),s=0.5,c='blue')
#plt.scatter(Jinford[0,:], Jinford[6,:],s=0.5,c='blue')

#plt.plot(Jinford[0,:], Jinford[6,:], drawstyle='steps-post', label='steps-post',linewidth=0.5,c='black')
#plt.ylim([0, 5])
plt.plot(Jinford[0,:], np.asarray(q0), drawstyle='steps-post', label='steps-post',linewidth=0.5,c='blue')
plt.plot(Jinford[0,:], np.asarray(q1), drawstyle='steps-post', label='steps-post',linewidth=0.5,c='red')

#plt.ylim([0, 5])
plt.plot(Jinford[0,:], -np.log(1-ecdf0(Jinford[0,:])),c='green')


# IF THERE IS BIAS IN THE HIDDEN LAYERS WE HAVE TO MULTIPLY BY Q0!!!!!

plt.savefig('../images/nonlinear.png', dpi=300)


plt.plot(Jinford[0,:], -np.log(1-ecdf0(Jinford[0,:])),c='green')
#plt.plot(Jinford[0,Jinford[7,:]==1], -np.log(1-ecdf1(Jinford[0,Jinford[7,:]==1])),c='orange')


# NEVER FORGET WE ARE ESTIMATING H_ZERO!!!!!!

plt.scatter(Jinford[0,:],Jinford[6,:],s=0.5,c='blue')
plt.scatter(Jinford[0,Jinford[7,:]==1],np.asarray(q),s=0.5,c='red')



plt.show()

model.mnonpar_val.jumps
model.final_lambda
model.hidden_dims
model.n_iters
model.patience