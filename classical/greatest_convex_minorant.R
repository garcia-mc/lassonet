library(fdrtool)

diagram <- read.table("~/lassoxnet/classical/diagram.txt", quote="\"", comment.char="")
x=c(0,as.numeric(diagram[1,]))
y=c(0,as.numeric(diagram[2,]))

plot(x,y)

gcm=gcmlcm(x, y, type=c("gcm"))
plot(x,y)
points(gcm$x.knots,gcm$y.knots,col='red')

derivatives=gcm$slope.knots

positions=which(x %in% gcm$x.knots) # position of the knots 

newLambda=numeric(length(x))
k=1

slopes=derivatives
for(i in 2:length(positions)) {
  while(k<=positions[i]) {
    newLambda[k]=slopes[i-1]
    k=k+1
  }
}
    
final=newLambda[-1]

# isto xa e a seguinte Lambda!!! gardar nun txt e pasar a python 
  
  

write.table(final, file = "lassoxnet/classical/gcmandslopes.txt", 
            sep = " ",row.names=FALSE,col.names=FALSE)