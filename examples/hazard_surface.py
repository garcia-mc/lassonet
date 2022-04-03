#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 14:06:39 2022

@author: carlos
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import random

def fun(x, y):
    predi=torch.zeros(len(x),X.shape[1])
    predi[:,8]=torch.from_numpy(x)
    predi[:,9]=torch.from_numpy(y)

    return np.exp(model.model.forward(predi).detach().numpy())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = var
Xm, Ym = np.meshgrid(x, y)
zs = np.array(fun(np.ravel(Xm), np.ravel(Ym)))
Z = zs.reshape(Xm.shape)

ax.plot_surface(Xm, Ym, Z)

ax.set_xlabel('Age')
ax.set_ylabel('Physical activity')
ax.set_zlabel('Hazard ratio')
ax.view_init(30, 230)
plt.savefig('../images/hazard.png', dpi=300)

plt.show()

