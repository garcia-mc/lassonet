# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np


class afunc:
  def __init__(self, delta1,delta2,delta3,u,v,z):
    self.delta1=delta1;
    self.delta2=delta2;
    self.delta3=delta3;
    self.u=u;
    self.v=v;
    self.z=z;

    self.Lambda=np.zeros(n);
  

  def etz(self,theta):
      return np.exp(np.dot(theta,self.z))
  def necox(self,theta):
      return np.exp(-self.Lambda*etz(theta))
  def a1(self,theta):
      return etz(theta)*necox(theta)/(1-necox(theta))
    
  def a2(self,theta):
      return etz(theta)*necox(theta)/(1-necox(theta))
    

p1 = Person("John", 36)
p1.myfunc() 

# to continue read just before proposition 2.1 in wellner
# is it true that we might not be able to use all the sample??? 
# or does he want to mean that we can randomly choose ESPECIFICALLY T(1) and T(m) (and let the others be the order statistics)
# discover it in groeneboom 1991