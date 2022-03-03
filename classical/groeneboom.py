# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

data=icgen()
class afunc:
  def __init__(self, indata,cov):
    
    self.u=indata[:,0];
    self.v=indata[:,1];
    self.delta1=indata[:,2];
    self.delta2=indata[:,3];
    self.delta3=indata[:,4];
    
    self.z=cov;
    
    self.n=len(self.u)

    self.Lambda=np.zeros(self.n);
    
    self.J1n=self.u[~ (self.delta3>0)]
    self.J2n=self.v[~ (self.delta1>0)]


  

  def etz(self,theta):
      return np.exp(np.dot(theta,self.z))
  def necox(self,theta):
      return np.exp(-self.Lambda*self.etz(theta))
  def a1(self,theta):
      return self.etz(theta)*self.necox(theta)/(1-self.necox(theta))
    
  def a2(self,theta):
      return self.etz(theta)*self.necox(theta)/(1-self.necox(theta))
    

res=afunc(data,0).J1n



# to continue read just before proposition 2.1 in wellner
# is it true that we might not be able to use all the sample??? 
# or does he want to mean that we can randomly choose ESPECIFICALLY T(1) and T(m) (and let the others be the order statistics)
# discover it in groeneboom 1991