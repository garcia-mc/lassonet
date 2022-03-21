# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt


data=icgen() # 999 means infinity

data=data.round(3)


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

    #self.Lambdau=np.log(1/(1-np.cumsum(np.random.uniform(epsilon,1-epsilon,n)/n)))
    #self.Lambdav=np.log(1/(1-self.Lambdau+epsilon/2))

    
    
    
  # def Lambda(self,border): # Cumulative hazard function 
  #       if border=='u':
  #           return self.Lambdau
  #       if border=='v':
  #           return self.Lambdav
    
  def etz(self,theta,iobs):
        return np.exp(np.matmul(self.z[int(iobs),:],theta))
  def necox(self,theta,Lambda,iobs): # Negative Exponential of Cox model
        
        return np.exp(-(Lambda*self.etz(theta,iobs)))
  def a1(self,theta,Lambda,iobs):
        return self.etz(theta,iobs)*self.necox(theta,Lambda,iobs)/(1-self.necox(theta,Lambda,iobs))
      
  def a2(self,theta,Lambdau,Lambdav,iobs):
        return self.etz(theta,iobs)*self.necox(theta,Lambdau,iobs)/(self.necox(theta,Lambdau,iobs)-self.necox(theta,Lambdav,iobs))
  
  def a3(self,theta,Lambdau,Lambdav,iobs):
        return self.etz(theta,iobs)*self.necox(theta,Lambdav ,iobs)/(self.necox(theta,Lambdau,iobs)-self.necox(theta,Lambdav,iobs))
      
  def W(self,t,theta):
          return(np.sum((self.delta1*self.a1(theta)-self.delta2*self.a2(theta))*(self.u<=t))+
          np.sum((self.delta2*self.a3(theta)-self.delta3*self.etz(theta))*(self.v<=t)))
  def G(self,theta):
          return(np.stack((self.delta1*(self.a1(theta))**2+self.delta2*(self.a2(theta))**2,
         self.delta2*(self.a3(theta))**2+self.delta3*(self.etz(theta))**2)))
      
#############

theta=np.zeros(2) 

z=np.random.rand(len(data),2)
model=afunc(data,z)




# one iteration        



model.J1n #these are all us
model.J2n #these are all vs

indu=np.squeeze(np.where(model.J1n >0))
indv=np.squeeze(np.where(model.J2n >0))
ind=np.concatenate((indu,indv),axis=0)
who=np.concatenate((np.repeat(2,len(indu)),np.repeat(3,len(indv))))

# 2 means u 3 means v

J=np.concatenate((model.J1n[indu],model.J2n[indv]))

m=len(J)

jdelta1=np.concatenate((model.delta1[indu],model.delta1[indv]))
jdelta2=np.concatenate((model.delta2[indu],model.delta2[indv]))
jdelta3=np.concatenate((model.delta3[indu],model.delta3[indv]))


Lambda0=np.log(1/(1-np.cumsum(np.ones(m)/(m+1))))
Jinfo=np.stack((J,ind,who,jdelta1,jdelta2,jdelta3,np.zeros(m)))

Jinford=Jinfo[:,np.argsort(J)]

# H=(lam/gamma)*(np.exp(gamma*Jinford[0,:])-1)*10


Jinford[6,:]=Lambda0
gcm=Lambda0

#################################3
nitergr=10
for i in range(nitergr):
    Gjumps=np.zeros(m)
    Wjumps=np.zeros(m)
    
    # row 1 is patient id
    # row 2 is border type
    # row 3 is delta1
    # row 4 is delta2
    # row 5 is delta3
    # row 6 is Lambda
    
    for k in range(m):
        
        if Jinford[2,k] == 2:
            if Jinford[3,k] !=0:
                Gjumps[k]=Gjumps[k] + model.a1(theta,Jinford[6,k],Jinford[1,k])**2
            if Jinford[4,k] !=0:
                Gjumps[k]=Gjumps[k] + model.a2(theta,Jinford[6,k],
                                               Jinford[6,np.logical_and(Jinford[1,:]==Jinford[1,k],Jinford[2,:]==3)],Jinford[1,k])**2
        if Jinford[2,k] == 3:
            
            if Jinford[4,k] !=0:
                Gjumps[k]=Gjumps[k] + model.a3(theta,
                                               Jinford[6,np.logical_and(Jinford[1,:]==Jinford[1,k],Jinford[2,:]==2)],Jinford[6,k],Jinford[1,k])**2
            
            if Jinford[5,k] !=0:
                Gjumps[k]=Gjumps[k] + model.etz(theta,Jinford[1,k])**2
    
        
    for k in range(m):
        
        if Jinford[2,k] == 2:
            if Jinford[3,k] !=0:
                Wjumps[k]=Wjumps[k] + model.a1(theta,Jinford[6,k],Jinford[1,k])
            if Jinford[4,k] !=0:
                Wjumps[k]=Wjumps[k] - model.a2(theta,Jinford[6,k],
                                               Jinford[6,np.logical_and(Jinford[1,:]==Jinford[1,k],Jinford[2,:]==3)],Jinford[1,k])
        if Jinford[2,k] == 3:
            
            if Jinford[4,k] !=0:
                Wjumps[k]=Wjumps[k] + model.a3(theta,
                                               Jinford[6,np.logical_and(Jinford[1,:]==Jinford[1,k],Jinford[2,:]==2)],Jinford[6,k],Jinford[1,k])
            
            if Jinford[5,k] !=0:
                Wjumps[k]=Wjumps[k] - model.etz(theta,Jinford[1,k])
    
    
    #ts=J
    
    
    # ts=np.linspace(4, 9, num=500)
    #Gvalues=np.zeros(m)
    W=np.zeros(m)
    G=np.zeros(m)
    integral=np.zeros(m)
    
    G=np.cumsum(Gjumps)
    W=np.cumsum(Wjumps)
    
    integral=np.cumsum(Jinford[6,:]*Gjumps)
    
    
    V=W+integral
    
    plt.plot(G, V)
    plt.show()
    plt.scatter(G,V ,s=0.1)
    plt.show()
    
    np.savetxt('diagram.txt', (G,V))
    gcm=robjects.globalenv['f'](robjects.FloatVector(G),robjects.FloatVector(V))

    
    Jinford[6,:]=gcm

plt.plot(Jinford[0,:], H)
plt.scatter(Jinford[0,:],gcm,s=0.5,c='black')
plt.show()


###################################################
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

gamma=5
lam=0.8

# ts=np.linspace(0, 10, num=50)
# h=lam*np.exp(gamma*ts)

# plt.plot(ts, h)
# plt.show()

H=(lam/gamma)*(np.exp(gamma*Jinford[0,:])-1)
# Hv=(lam/gamma)*(np.exp(gamma*model.v)-1)

plt.plot(Jinford[0,:], H)
plt.scatter(Jinford[0,:],gcm,s=0.5,c='black')
plt.show()

linitu=