import math
import phd_csv
import numpy as np
from matplotlib import pyplot as plt 
import gpmodel 
import gpkernelparams as gpk

k ,data = phd_csv.pulse_guide()
lb = 3000 
ub = 4500

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
freq = np.fft.fftfreq(t.shape[-1])
low = 5
up = 3*len(freq)//64
sp = np.fft.fft(observed)
ps = np.power(np.abs(sp),2)
"""
plt.figure(1)
plt.plot(t,observed)
plt.figure(2)
plt.plot(freq[low:up],sp.real[low:up])
plt.figure(3)
plt.plot(freq[low:up],sp.imag[low:up])
plt.figure(4)
plt.plot(freq[low:up],ps[low:up])
plt.show()
"""
new = np.linspace(t[0],t[-1]+ 200,2*n)
#print(t[0] -t[-1],n)
noise = 0.05
gpk.figures_kernels(new,t, observed, noise)



"""
i=0
lmse = 10
for i in range(600):
    lmse, params = gpmodel.csv_gp(t,a, lmse,100, i1 = 0 + i, i2 =200 + i, N= 16, plot= True)
    print(lmse)
    print(params)
"""
"""
plt.subplot(3,1,1)
plt.plot(t,part[:,3]*1000,"o")
plt.plot(t,a,"+")
plt.plot(t,u,"ro")
plt.subplot(3,1,2)
plt.plot(t,a,"o")
#plt.plot(t,dither)
#plt.plot(t,y-p,"+")
plt.subplot(3,1,3)
plt.plot(t,u,"+")
plt.show()
"""


