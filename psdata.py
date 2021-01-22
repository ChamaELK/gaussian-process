import math
import phdcsv
import numpy as np
from matplotlib import pyplot as plt 
import gpmodel 
import gpkernelparams as gpk


def cumsum(lb,ub):
    k ,data = phdcsv.pulse_guide()
    part = data[lb:ub]
    u = part[:,1]
    p = part[:,1]
    p[np.where(np.isnan(p))]=0
    y = np.array(part[:,3]*1000)
    t = part[:,0]
    dither = part[:,5]
    n=len(t)
    i=0
    j=0
    dtlimit = 0.2
    a = np.zeros(n)
    while i<n :
        j=i 
        if math.isnan(y[i]) :
            i-=1
        while math.isnan(y[j]) :
            if j<n-1 :
                j+=1
        if j>i:
            for k in range(i+1,j):
                y[k]= y[i] +((t[k]-t[i])*(y[j]-y[i])/(t[j]-t[i]))
            i=j 
        i+=1

    a= y-p

    for i in range(n-2):
        if any(p[i:i+2]!=0): 
            if t[i+1]-t[i] < dtlimit :
                a[i]= y[i] - sum(p[i:i+2])


    # observed data 
    np.append(0,p)
    observed = np.cumsum(p) + a
    return t, observed


