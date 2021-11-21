# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:00:25 2021

@author: Carlos M Garcia
"""

output=model(X_train[batch]) #will act as betaz

def lox(betaz, times, deltas):
    n=len(deltas)
    M=np.zeros((n,n))
    ebetaz=torch.exp(betaz)
    for i in range(n):
        M[i,:]=np.greater_equal(times,times[i])*1
    Mt=torch.from_numpy(M)
    return -torch.sum((betaz-torch.matmul(Mt,ebetaz))*torch.from_numpy(deltas))


model = nn.Linear(2, 2)
x = torch.randn(1, 4)
target = torch.randn(1, 2)
output = model(x)
loss = my_loss(output, target)
loss.backward()
print(model.weight.grad)

   batch = indices[0:len(x)]
   
   
   #H censor indicator
   #i is lifetime
   #t is age
   #u bmi
   #W 
   #ai presion arterial
   #da cc á bi
   # cal é a mellor metrica de acelerometros 
   