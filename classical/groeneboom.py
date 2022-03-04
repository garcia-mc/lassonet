# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

data=icgen() # 999 means infinity
class afunc:
  def __init__(self, indata,cov):
    
    self.u=indata[:,0];
    self.v=indata[:,1];
    self.delta1=indata[:,2];
    self.delta2=indata[:,3];
    self.delta3=indata[:,4];
    
    self.z=cov;
    
    self.n=len(self.u)
    
    self.J1n=self.delta1*self.u+self.delta2*self.u
    self.J2n=self.delta3*self.v+self.delta2*self.v

    self.Lambdau=self.J1n+0.1;
    self.Lambdav=self.J2n+0.2;

    
    
    
  def Lambda(self,border): # Cumulative hazard function 
        if border=='u':
            return self.Lambdau
        if border=='v':
            return self.Lambdav
    
  def etz(self,theta):
        return np.exp(np.matmul(self.z,theta))
  def necox(self,theta,border): # Negative Exponential of Cox model
        
        return np.exp(-(self.Lambda(border)*self.etz(theta)))
  def a1(self,theta):
        return self.etz(theta)*self.necox(theta,'u')/(1-self.necox(theta,'u'))
      
  def a2(self,theta):
        return self.etz(theta)*self.necox(theta,'u')/(self.necox(theta,'u')-self.necox(theta,'v'))
  
  def a3(self,theta):
        return self.etz(theta)*self.necox(theta,'v')/(self.necox(theta,'u')-self.necox(theta,'v'))
      
  def W(self,t,theta):
          return(np.sum((self.delta1*self.a1(theta)-self.delta2*self.a2(theta))*(self.u<=t))+
          np.sum((self.delta2*self.a3(theta)-self.delta3*self.etz(theta))*(self.v<=t)))
  def G(self,t,theta):
          return(np.sum((self.delta1*(self.a1(theta))**2+self.delta2*(self.a2(theta))**2)*(self.u<=t))+
          np.sum((self.delta2*(self.a3(theta))**2+self.delta3*(self.etz(theta))**2)*(self.v<=t)))

  
z=np.random.rand(len(data),2)
model=afunc(data,z)
model.Lambda(border='v')
model.W(1,theta=np.zeros(2))

model.J1n #these are all us
model.J2n #these are all vs

J=np.union1d(model.J1n,model.J2n)

ts=J
m=len(ts)
Gvalues=np.zeros(m)
Wvalues=np.zeros(m)

for i in range(m):
    Gvalues[i]=model.G(ts[i],theta=np.zeros(2))
for i in range(m):
    Wvalues[i]=model.W(ts[i],theta=np.zeros(2))

model.a1(theta=np.zeros(2))


import matplotlib.pyplot as plt

# we need to do ts = u union v to be able to integrate 
plt.plot(ts, Wvalues)
plt.show()

np.diff(Gvalues)

# crear o famoso vector Jn acompanhado doutros dous 
# un que indique o individuo do que proven
# outro se e unha u ou unha v para ter acceso
# to continue read just before proposition 2.1 in wellner
# is it true that we might not be able to use all the sample??? 
# or does he want to mean that we can randomly choose ESPECIFICALLY T(1) and T(m) (and let the others be the order statistics)
# discover it in groeneboom 1991