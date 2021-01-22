import psdata


observed = psdata.cumsum(3000,4500)
freq = np.fft.fftfreq(t.shape[-1])
low = 5
up = 3*len(freq)//64
sp = np.fft.fft(observed)
ps = np.power(np.abs(sp),2)

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
#new = np.linspace(t[0],t[-1]+ 200,2*n)
#print(t[0] -t[-1],n)
#noise = 0.05
#gpk.figures_kernels(new,t, observed, noise)

i=0
lmse = 10
for i in range(600):
    lmse, params = gpmodel.csv_gp(t,a, lmse,100, i1 = 0 + i, i2 =200 + i, N= 16, plot= True)
    print(lmse)
    print(params)

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

