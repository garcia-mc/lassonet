# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt


data=icgen() # 999 means infinity

data=data.round(3)

epsilon=0.05

n=len(data)



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

    self.Lambdau=np.log(1/(1-np.cumsum(np.random.uniform(epsilon,1-epsilon,n)/n)))
    self.Lambdav=np.log(1/(1-self.Lambdau+epsilon/2))

    
    
    
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
        return self.etz(theta)*self.necox(theta,'u')/np.max(epsilon,(1-self.necox(theta,'u')))
      
  def a2(self,theta):
        return self.etz(theta)*self.necox(theta,'u')/np.max(epsilon,(self.necox(theta,'u')-self.necox(theta,'v')))
  
  def a3(self,theta):
        return self.etz(theta)*self.necox(theta,'v')/np.max(epsilon,(self.necox(theta,'u')-self.necox(theta,'v')))
      
  def W(self,t,theta):
          return(np.sum((self.delta1*self.a1(theta)-self.delta2*self.a2(theta))*(self.u<=t))+
          np.sum((self.delta2*self.a3(theta)-self.delta3*self.etz(theta))*(self.v<=t)))
  def G(self,theta):
          return(np.stack((self.delta1*(self.a1(theta))**2+self.delta2*(self.a2(theta))**2,
         self.delta2*(self.a3(theta))**2+self.delta3*(self.etz(theta))**2)))
      
#############
z=np.random.rand(len(data),2)
model=afunc(data,z)

def Wnew

# one iteration        

plt.scatter(model.u, model.Lambda('u'))
plt.show()


model.Lambda(border='v')
Gvalues=np.zeros((2,n))
Gvalues=model.G(theta=np.zeros(2))

model.J1n #these are all us
model.J2n #these are all vs

indu=np.squeeze(np.where(model.J1n >0))
indv=np.squeeze(np.where(model.J2n >0))
ind=np.concatenate((indu,indv),axis=0)
who=np.concatenate((np.repeat('u',len(indu)),np.repeat('v',len(indv))))
J=np.concatenate((model.J1n[indu],model.J2n[indv]))

Jinfo=np.stack((J,ind,who))


ts=J

J=np.sort(J)

# ts=np.linspace(4, 9, num=500)
m=len(ts)
#Gvalues=np.zeros(m)
Wvalues=np.zeros(m)
G=np.zeros(m)
integral=np.zeros(m)

for i in range(m):
    G[i]=np.sum(Gvalues[0,]*(model.u<=J[i]))+np.sum(Gvalues[1,]*(model.v<=J[i]))

for i in range(m):

    integral[i]=np.sum(model.Lambda(border='u')*Gvalues[0,]*(model.u<=J[i]))+np.sum(model.Lambda(border='v')*Gvalues[1,]*(model.v<=J[i]))

for i in range(m):
    Wvalues[i]=model.W(J[i],theta=np.zeros(2))

V=Wvalues+integral

plt.plot(G, V)
plt.show()
plt.scatter(G,V ,s=0.1)
plt.show()

np.savetxt('diagram.txt', (G,V))
gcm=np.loadtxt('gcmandslopes.txt')

# reconstruir lambda co que viu de gcm 

plt.scatter(J, G,s=0.1)
plt.show()

np.diff(Gvalues)

# crear o famoso vector Jn acompanhado doutros dous 
# un que indique o individuo do que proven
# outro se e unha u ou unha v para ter acceso
# to continue read just before proposition 2.1 in wellner
# is it true that we might not be able to use all the sample??? 
# or does he want to mean that we can randomly choose ESPECIFICALLY T(1) and T(m) (and let the others be the order statistics)
# discover it in groeneboom 1991

# gompertz survival function

gamma=0.1
lam=0.1

ts=np.linspace(0, 10, num=50)
h=lam*np.exp(gamma*ts)

plt.plot(ts, h)
plt.show()

H=(lam/gamma)*(np.exp(gamma*ts)-1)
Hv=(lam/gamma)*(np.exp(gamma*model.v)-1)

plt.plot(ts, H)
plt.show()

linitu=