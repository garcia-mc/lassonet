#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:47:16 2022

@author: duser1
"""

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

model = torch.nn.Linear(1, 1)
loss = my_loss(output, target)
loss.backward()


x = torch.randn(30, 1)
target = 2.7*x
output = model(x)

	pred_y = our_model(xl)
    

a = torch.ones(30,1)*0.5

torch.bernoulli(a)


##########
    
class LinearRegressionModel(torch.nn.Module):

	def __init__(self):
		super(LinearRegressionModel, self).__init__()
		self.linear = torch.nn.Linear(1, 1,bias=False) # One in and one out

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred
    
    # our model
our_model = LinearRegressionModel()

criterion = model.loss_wellner
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

for epoch in range(200):

	# Forward pass: Compute predicted y by passing
	# x to the model
	pred_y = our_model(zt)

	# Compute and print loss
	loss = criterion(pred_y)

	# Zero gradients, perform a backward pass,
	# and update the weights.
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print('epoch {}, loss {}'.format(epoch, loss.item()))

for param in our_model.parameters():
    print(param)

