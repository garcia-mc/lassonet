#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:58:20 2022

@author: carlos
"""
Jinford=model.mnonpar_full.JJ

Xnp=X.numpy()

plt.hist()
np.min(Xnp[:,8])
# variables [bmi,cancer,stroke,diabetes,one_hot,X]
# age is first continuous: 8
# tac is the one after age: 9
nplots=20
var=torch.linspace(-1.5,2,nplots)
predi0=torch.zeros(len(Jinford[6,:]),X.shape[1])
np.min(Xnp[:,9])
predi0[:,9]=torch.ones(len(predi))*np.min(Xnp[:,9])


q0=np.multiply(np.asmatrix(Jinford[6,:]).transpose(),np.exp(model.model.forward(predi0).detach().numpy()))
plt.plot(Jinford[0,:], np.exp(-np.asarray(q0)), drawstyle='steps-post', label='steps-post',linewidth=0.5,c='black')
valores=[]

for i in range(nplots):
    predi=torch.zeros(len(Jinford[6,:]),X.shape[1])
    predi[:,9]=torch.ones(len(predi))*var[i]
    c = cm.Greens(i/nplots,1)
    q1=np.multiply(np.asmatrix(Jinford[6,:]).transpose(),np.exp(model.model.forward(predi).detach().numpy()))
    plt.plot(Jinford[0,:], np.exp(-np.asarray(q1)), drawstyle='steps-post', label='steps-post',linewidth=0.5,c=c)
    valores.append(np.exp(model.model.forward(predi).detach().numpy())[0])
    

plt.xlabel('Time', fontsize=12)
plt.ylabel('Survival function', fontsize=12)

plt.savefig('../images/nonlinear_surv_tac.png', dpi=300)

plt.plot(var,np.log(valores),c='black')

plt.xlabel('TAC', fontsize=12)
plt.ylabel('Nonlinear Cox regression function', fontsize=12)

plt.savefig('../images/nonlinear_surv_tac_func.png', dpi=300)


#plt.plot(Jinford[0,:], H0,c='grey')
#plt.scatter(Jinford[0,:],np.asarray(q0),s=0.5,c='blue')
#plt.scatter(Jinford[0,:], Jinford[6,:],s=0.5,c='blue')

#plt.plot(Jinford[0,:], Jinford[6,:], drawstyle='steps-post', label='steps-post',linewidth=0.5,c='black')
#plt.ylim([0, 5])
plt.plot(Jinford[0,:], np.asarray(q0), drawstyle='steps-post', label='steps-post',linewidth=0.5,c='black')


import matplotlib.cm as cm

cm.Greens(10)