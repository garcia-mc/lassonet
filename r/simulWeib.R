# from https://stats.stackexchange.com/questions/135124/how-to-create-a-toy-survival-time-to-event-data-with-right-censoring

# baseline hazard: Weibull

# N = sample size    
# lambda = scale parameter in h0()
# rho = shape parameter in h0()
# beta = fixed effect parameter
# rateC = rate parameter of the exponential distribution of C

simulWeib <- function(N, lambda, rho, beta, rateC)
{
  p=length(beta)
  X<- matrix(runif(n = N*p,min=0,max=1),ncol=p)
  # covariate --> N Bernoulli trials
  # x <- sample(x=c(0, 1), size=N, replace=TRUE, prob=c(0.5, 0.5))
  #x= runif(N,0,1)
  # Weibull latent event times
  v <- runif(n=N)
  Tlat <- (- log(v) / (lambda * exp(X%*%beta)))^(1 / rho)
  
  # censoring times
  
  status=1-rbinom(N,1,rateC) # 1 is not censored
  time=Tlat-(1-status)*runif(N,min=rep(0,N),max=Tlat/2)

  
  # data set
  cbind(as.data.frame(X),data.frame(id=1:N,
                                    time=time,
                                    status=status))
}
