library(survival)

p=3
B=1000
set.seed(1234)
betaHat <- matrix(0,nrow=1000,ncol=p)
for(k in 1:B)
{
  dat <- simulWeib(N=2000, lambda=0.01, rho=1, beta=c(-0.6,0.4,1), rateC=0.3)
  fit <- coxph(Surv(time, status) ~ V1 + V2 + V3, data=dat)
  betaHat[k,] <- fit$coef
}

apply(betaHat,2,mean)
