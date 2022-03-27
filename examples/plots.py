#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:29:58 2022

@author: carlos
"""
plt.plot(Jinford[0,:], np.exp(-H0),c='grey')
#plt.scatter(Jinford[0,:],np.asarray(q0),s=0.5,c='blue')
#plt.scatter(Jinford[0,:], Jinford[6,:],s=0.5,c='blue')

plt.plot(Jinford[0,:], np.exp(-Jinford[6,:]), drawstyle='steps-post', label='steps-post',linewidth=0.5,c='black')
#plt.ylim([0, 5])

plt.xlabel('Time', fontsize=12)
plt.ylabel('Baseline survival function', fontsize=12)

plt.savefig('../images/nonlinear_surv.png', dpi=300)
