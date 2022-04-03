#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:58:20 2022

@author: carlos
"""
import matplotlib.cm as cm

Jinford=model.mnonpar_full.JJ


# variables [gender,cancer,stroke,diabetes,one_hot,X]
# age is first continuous: 8
# tac is the one after age: 9
# bmi is last one 29
nplots=10
var=torch.linspace(0,1,nplots)
predi0=torch.zeros(len(Jinford[6,:]),X.shape[1])

select=9


q0=np.multiply(np.asmatrix(Jinford[6,:]).transpose(),np.exp(model.model.forward(predi0).detach().numpy()))
plt.plot(Jinford[0,:], np.exp(-np.asarray(q0)),'k-', drawstyle='steps-post', label='steps-post',linewidth=0.5)
valores=[]

for i in range(nplots):
    if i==0: 
        continue
    predi=torch.zeros(len(Jinford[6,:]),X.shape[1])
    predi[:,select]=torch.ones(len(predi))*var[i]
    c = cm.Greens(var[i].numpy(),1)
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

### discrete

predi=torch.zeros(len(Jinford[6,:]),X.shape[1])
predi0=torch.zeros(len(Jinford[6,:]),X.shape[1])
predi[:,1]=torch.ones(len(predi))
q0=np.multiply(np.asmatrix(Jinford[6,:]).transpose(),np.exp(model.model.forward(predi0).detach().numpy()))
q1=np.multiply(np.asmatrix(Jinford[6,:]).transpose(),np.exp(model.model.forward(predi).detach().numpy()))



plt.plot(Jinford[0,:], np.exp(-np.asarray(q0)), drawstyle='steps-post', label='steps-post',linewidth=0.5,c='black')
plt.plot(Jinford[0,:], np.exp(-np.asarray(q1)), drawstyle='steps-post', label='steps-post',linewidth=0.5,c='green')



#plt.plot(Jinford[0,:], H0,c='grey')
#plt.scatter(Jinford[0,:],np.asarray(q0),s=0.5,c='blue')
#plt.scatter(Jinford[0,:], Jinford[6,:],s=0.5,c='blue')

#plt.plot(Jinford[0,:], Jinford[6,:], drawstyle='steps-post', label='steps-post',linewidth=0.5,c='black')
#plt.ylim([0, 5])
plt.plot(Jinford[0,:], np.asarray(q0), drawstyle='steps-post', label='steps-post',linewidth=0.5,c='black')



cm.Greens(10)