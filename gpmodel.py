import numpy as np
import math 
import matplotlib.pyplot as plt
import phdcsv
from scipy.optimize import minimize
from numpy.linalg import cholesky, det, lstsq
import mpc
 
class Kernels():
    #def __init__(self):

    #Kernel
    # covariance functions
    def kse(self,t1,t2, lse):
        return math.exp(-(t1-t2)**2/(2*lse))

    def dkse_dlse(self,t1,t2,lse):
        return ((t1-t2)**2/(lse**2))*math.exp(-(t1-t2)/(2*lse))
    def kp(self,t1,t2, lp, u):
        s= math.sin(math.pi*(t1-t2)/u)
        return math.exp(-2*s**2/(lp**2))

    def kc(self,t1,t2,th,lse,lp,u):
        return (th**2)*self.kse(t1,t2,lse)*self.kp(t1,t2, lp, u)

    def dkp_du(self,t1,t2,lp,u):
        s = math.sin(math.pi*(t1-t2)/u)
        ds_du = 4*(t1-t2)*math.pi*s /(u**2*lp**2)
        return  math.exp(-2*s**2/(lp**2))
    def dkc_du(self,t1,t2,th,lse,lp,u):
        return (th**2)*self.kse(t1,t2,lse)*self.dkp_du(t1,t2,lp,u)
     
class GProcess():
  
    def __init__(self,kernel,kernel_derivate,**kernel_params):
        self.kernel= kernel
        self.kernel_params= kernel_params
        self.kernel_derivate = kernel_derivate
    def kernel_modify(self,keys,values):
        if len(keys) == len(values) :
            for k,v in zip(keys,values):
                self.kernel_params[k] = v
    # cross covariance between the  observed inputs and the new input
    # new : dimension n 
    #observed: dimension m
    def cov( self, x, y,order = 0):
        n = len(x)
        m = len(y)
        kernel= self.kernel
        if order == 1:
            kernel = self.kernel_derivate
        ccov= np.zeros(shape = (n,m))
        for i in range(n):
            for j in range(m):
                ccov[i,j] = kernel(x[i],y[j],**self.kernel_params)
        return ccov
    

    # parameters:  observed output and noise
    #predictive mean and variance of a new output 
    # kernel : type of kernel in use
    def predict(self,new, observed, output, noise):
        m= len(observed)
        sigma= noise**2 * np.eye(m)
        # K covariance matrix
        K= self.cov(observed,observed) 
        # Kno: cross covariance new observed
        Kno= self.cov(new,observed)
        A= np.linalg.inv(K + sigma )
        # mean
        mean= Kno.dot(A).dot(output)
        # variance
        Knn= self.cov(new,new)
        variance= Knn - Kno.dot(A).dot(Kno.T)
        return mean, variance
        
    def ll_naive(self,observed, output, noise, keyparams, values):
        self.kernel_modify(keyparams,values)
        m= len(observed)
        s = noise**2
        sigma= s*np.eye(m)
        K= self.cov(observed,observed) + sigma
        log_likelihood =  0.5 * math.log(np.linalg.det(K)) + \
               0.5 * output.T.dot(np.linalg.inv(K).dot(output)) + \
               0.5 * len(observed) * math.log(2*math.pi)    
        return log_likelihood
    def ll_stable(self,observed,output,noise, keyparams,values):
        self.kernel_modify(keyparams,values)
        m= len(observed)
        s = noise**2
        sigma= s*np.eye(m)
        K= self.cov(observed,observed) + sigma  
        L = cholesky(K)
        return sum(np.log(np.diagonal(L))) + \
            0.5 * output.T.dot(lstsq(L.T, lstsq(L, output)[0])[0]) + \
            0.5 * len(observed) * math.log(2*math.pi)

    def get_ll(self,ll_type):
        if ll_type == 0:
            return self.ll_naive
        if ll_type == 1 :
            return self.ll_stable

    def lml(self,observed, output, noise, params, ll_type= 1):    
        ll = self.get_ll(ll_type)
        keyparams = params.keys()
        start = list(params.values())
        f = lambda hyper :ll(observed,output,noise,keyparams,hyper) 
        res = minimize(f,start,method='L-BFGS-B')
        logl= res.fun
        #f = lambda keys,hyper : ll(observed,output,noise,keys,hyper)
        #self.ll_plot(f,parambounds.key(),parambounds.values())
        return res.x, logl

    def ll_plot(self, observed,output,noise, keys, bounds,n,dim =1, ll_type= 1):
        ll = self.get_ll(ll_type)
        if len(keys) == 1 :
            x = np.logspace(bounds[0][0], bounds[0][1], n)
            y = np.zeros(n)
            for i in range(n):
                y[i] = ll(observed,output,noise,keys,[x[i]])
            plt.plot(x,y)
        if len(keys) >= 2 : 
            x = np.logspace(bounds[0][0], bounds[0][1], n)
            y = np.logspace(bounds[1][0], bounds[1][1], n)
            Z = np.zeros((n,n))
            X, Y = np.meshgrid(x,y)
            print(keys[:2])
            for i in range(n):
                for j in range(n):
                    Z[i,j] = ll(observed,output,noise,keys[:2],[X[i,j],Y[i,j]])
            plt.contour(X,Y,Z,500)

    def dll_du(self, u, observed, output, noise):
        self.kernel_params['u'] = u     
        m = len(observed)
        s = noise**2
        sigma= s*np.eye(m)
        K = self.cov(observed,observed) + sigma
        L = cholesky(K)  
        a = L.dot(L.T).dot(output)
        dll_du = 0.5* np.trace((a.dot(a.T) - L.dot(L.T)).dot(self.cov(observed, observed, 1))) 
        return dll_du

    def dlloo(self,u,observed,output,noise):
        result = 0
        self.kernel_params['u'] = u     
        m = len(observed)
        s = noise**2
        sigma= s*np.eye(m)
        K = self.cov(observed,observed) + sigma 
        iK = np.linalg.inv(K)
        a = iK.dot(output)
        Z = iK.dot(self.dll_du(u,observed,output,noise))
        for i in range(m):
            result = (a[i]*(Z.dot(a)[i]) - (0.5*(1+(a[i]**2/iK[i,i]))*Z.dot(iK)[i,i]))/iK[i,i]
        return result
    def optimize(self, u, observed, output, noise):
        umin = 2
        umax = 600
        
        min_du = 1e-2
        partial_u = np.zeros(2)
        partial_u[0] = self.dlloo(u,observed, output ,noise) 
        #partial_u[1] = self.dlloo(u-u/2,observed,output,noise)
        du = - u/2
        u += du
        rprop = Rprop(partial_u[0])   
        while du>min_du and u>= umin and u <= umax :
            partial_u[1] = self.dll_du(u,observed, output ,noise)
            rprop.update(partial_u)
            partial_u[0] = partial_u[1]
            du = np.sign(partial_u[1]) * rprop.g
            u -= du
            print(u) 
class Rprop():
    ## pfac y nfac refers to positive and negative factor 
    ## gmax gmin are max and min values of the gradient
    def __init__(self,g0, pfac= 1.2, nfac= 0.9 ,gmax = 50,gmin = 1e-5 ):
        self.g = g0
        self.pfac = pfac
        self.nfac = nfac
        self.gmax = gmax
        self.gmin = gmin
    def update(self,partial):
        psign = np.sign(np.prod(partial))
        if psign == 1:
            self.g = min(self.g * self.pfac, self.gmax)
        if psign == -1:
            self.g = max(self.g * self.nfac, self.gmin)


def test():      
    new= np.arange(9.5,12,0.5)
    n= len(new)
    observed=np.arange(-10,10,0.5) 
    m= len(observed)
    output= 5*np.sin(observed) + 0.1*np.random.randn(m)
    noise= 0.1* np.ones(len(observed))
    # lse = 5 , lp = 2, u = 10, th= 5 
    params = { "lse" : 5 , "lp" : 2, "u" : 10, "th": 5}
    gp= GProcess(kernel= Kernels().kc, th=5,lse= 5,lp=2,u=10)
    m, var= gp.predict(new,observed, output, noise)
    plot_gp(m,var,new,observed,output)

def plot_gp(m,var,new,observed,output,true_output,filename):
    uncertainty = 1.96 * np.sqrt(np.diag(var))
    plt.plot(observed,output,'ro')
    plt.plot(new,m,'ro',color="blue")
    plt.plot(new,true_output,'ro',color='green')
    plt.fill_between(new,m+uncertainty, m- uncertainty, alpha=0.1)
    plt.savefig(filename)
    plt.clf()
    #plt.show()

def csv_gp(t, ar, lmse, nu, i1 , i2 , N , plot= False):
    gp = GProcess(Kernels().kc,Kernels().dkc_du, u = 300, lse = 1200, th = 100, lp= 1.7)
    i3 = i2 + N
    noise=np.mean(lmse) 
    m, var = gp.predict(t[i2:i3],t[i1:i2],ar[i1:i2], noise )
    s2 = np.linalg.norm(np.linalg.matrix_power(var + noise**2*np.eye(N),2))
    lmse = 0.5*math.log(2*math.pi*s2) + (ar[i2-N:i2]-m)**2/(2*s2)
    if plot:
        plot_gp(m,var,t[i2:i3],t[i1:i2],ar[i1:i2],ar[i2:i3], "figures/gp_20201030T043254/"+str(i1))
    params = gp.kernel_params
    return lmse , params
def csv_gp_mpc_tracking():
  t, dither,  ur, ud, xr, xd = phdcsv.ra_dec_data(plot= False)
  ar = xr- ur

  # MPC Params 
  A = 1 
  B = 1
  C = 1
  Q = 100
  R = 2 
  S = 0.00001

  i1 = 50
  i2 =150
  N= 15
  M = 10
  x = np.zeros(M)
  u = np.zeros(M)
  r = np.zeros(M)
  err1 = np.zeros(M+1)
  err2 = np.zeros(M+1)
  for i in range(M):
    m, var  = csv_gp(t,ar,err2[i],i1,i2,N)
    ref = np.zeros(N)
    ref[0]= m[0]
    r[i] = m[0]
    for j in range(1,N):
      ref[j] += m[j-1]
    x0 = ar[i2-1]
    du , xt = mpc.mpc_tracking(N,x0,ref,A,B,C,Q,R,S)
    u[i] = xt
    noise= 0.1* np.ones(len(observed))
    # lse = 5 , lp = 2, u = 10, th= 5 
    params = { "lse" : 5 , "lp" : 2, "u" : 10, "th": 5}
    gp= GProcess(kernel= Kernels().kc, th=5,lse= 5,lp=2,u=10)
    m, var= gp.predict(new,observed, output, noise)
    plot_gp(m,var,new,observed,output)

    x[i] = xt[0]
    err1[i+1] = x[i]- ar[i2]
    err2[i+1] = ar[i2] -m[0]
    i1 += 1
    i2 += 1
  
  print("x")
  print(x)
  print("u")
  print(u)
 
  plt.subplot(3,1,1)
  plt.bar(np.arange(M),u)
  plt.subplot(3,1,2)
  plt.plot(x)
  plt.plot(r)
  plt.subplot(3,1,3)
  plt.plot(err1)
  plt.plot(err2)
  plt.show()

#test()
"""
t, dither,  ur, ud, xr, xd = phd_csv.ra_dec_data(filename = "cvastrophoto_guidelog_20201006T013332.txt",plot= False)
ar = xr- ur
i=0
lmse = 30
for i in range(40):
    lmse, params = csv_gp(t,ar, lmse,100, i1 = 0 + i, i2 =50 + i, N= 4, plot= True)
    print(lmse)
    print(params)
#csv_gp_mpc_tracking()

"""
