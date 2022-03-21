#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:50:39 2022

@author: carlos
"""

def R():
    from rpy2 import robjects
    
    from rpy2.robjects.packages import importr
    # imports the base module for R.
    base = importr("base")
     
    # imports the utils package for R.
    utils = importr("utils")
    
    # import fdrtool
    
    utils = importr("fdrtool")
    
    gcmpy = robjects.r['gcmlcm']
    
    
    
    
    
    robjects.r('''
    f <- function(G,V) {
        # library(fdrtool)
    
        x=c(0,G)
        y=c(0,V)
        
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
        return(final)
    }
    
    ''')
    



