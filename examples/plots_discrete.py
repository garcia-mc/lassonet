#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:42:06 2022

@author: carlos
"""
plt.plot(Jinford[0,Jinford[7,:]==0], np.exp(-H0),c='grey')
plt.plot(Jinford[0,Jinford[7,:]==1], np.exp(-H1),c='grey')



q=np.multiply(np.asmatrix(Jinford[6,Jinford[7,:]==1]).transpose(),np.exp(model.model.forward(Xt1[Jinford[1,Jinford[7,:]==1].astype(int),:]).detach().numpy()))
q0=np.multiply(np.asmatrix(Jinford[6,Jinford[7,:]==0]).transpose(),np.exp(model.model.forward(Xt0[Jinford[1,Jinford[7,:]==0].astype(int),:]).detach().numpy()))
# NEVER FORGET WE ARE ESTIMATING H_ZERO!!!!!!

plt.plot(Jinford[0,Jinford[7,:]==0],np.exp(-np.asarray(q0)),c='green',drawstyle='steps-post', label='steps-post',linewidth=0.75)
plt.plot(Jinford[0,Jinford[7,:]==1],np.exp(-np.asarray(q)),c='red',drawstyle='steps-post', label='steps-post',linewidth=0.75)



plt.xlabel('Time', fontsize=12)
plt.ylabel('Survival functions', fontsize=12)

plt.savefig('../images/nonlinear_surv.png', dpi=300)
