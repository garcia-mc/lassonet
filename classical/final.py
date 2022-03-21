#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:54:30 2022

@author: duser1
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

###

from preproc import preproc
from L import L
from cox_gen import cox_gen

# z=np.random.rand(500,3)

# z=np.random.binomial(1,0.5,[500,1])

a = torch.ones(300,1)*0.5

zt=torch.bernoulli(a)

z=zt.numpy()

data=cox_gen(z,[3.1]) # 999 means infinity

data=data.round(5)

res=preproc(data)

#############


model=L(res,z)
comprobar=model.JJ


theta=[0.1] # try to change it, final plot will only work with theta = 0.7


model.fit(theta,20)

Jinford=model.JJ

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

######
n=len(data)
thetas=np.linspace(0.5,4,num=100)
likes=np.zeros(100)

for l in range(100):
    thet=thetas[l]
    ln=np.zeros(n)
    m=thet*zt
    likes[l]=model.loss_wellner(m)


plt.scatter(thetas,likes)
plt.show()
    
thetas[np.argmin(likes)]

##### Now with pytorch 

class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(1, 1,bias=False) # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred
    
our_model = LinearRegressionModel()

optimizer = torch.optim.SGD(our_model.parameters(), lr = 2)

for param in our_model.parameters():
    print(param)
    thetaiter=param


### fitting several nn

for outers in range(10):
    with torch.no_grad():
        th=thetaiter.clone().numpy()
        model.fit(th,20)
    criterion = model.loss_wellner
    for epoch in range(10):
        pred_y = our_model(zt)
        
    
        loss = criterion(pred_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))
        for param in our_model.parameters():
            print(param)
            thetaiter=param


# ### now changing the loss dynamically


# for epoch in range(200):
#     with torch.no_grad():
#         th=thetaiter.clone().numpy()
#         model.fit(th)
#     criterion = model.loss_wellner
#     pred_y = our_model(zt)
    

#     loss = criterion(pred_y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print('epoch {}, loss {}'.format(epoch, loss.item()))
#     for param in our_model.parameters():
#         print(param)
#         thetaiter=param




# model.fit(thetaiter)
