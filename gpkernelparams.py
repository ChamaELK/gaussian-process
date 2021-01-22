import gpmodel as gp
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import collections



def figures_kernels(new,observed,output, noise):
    samples = 5
    k = [gp.Kernels().kse,gp.Kernels().kp,  gp.Kernels().kc]
    dk = [gp.Kernels().dkse_dlse,gp.Kernels().dkp_du, gp.Kernels().dkc_du]
    params= [{"lse" : [0.5,1]},{"lp":[1,3],"u":[0.5,1]},{"lse" :[0.5,1], "lp":[1,3],"th":[0.5,2],"u":[0.5,2.5]}]
    ll_bounds = [[{"lse" : [1,3]}],[{"lp" : [1,3],"u": [1,3]}],[{"th":[1,3], "u":[1,3]},{"lse" : [1,3], "u": [0.5,3]} ]] 
    nvalues = 2
    for i in range(len(params)):
        kp = [{key : value[j] for key, value in params[i].items()} for j in range(nvalues)] 
        figures_k(new,observed,output,noise,samples,k[i],dk[i],kp, ll_bounds[i])
    

def figures_k(new,observed, output, noise,samples,k,dk,kp, bounds):
    gpr = gp.GProcess(k, dk)
    gpr.kernel_params =kp[0]
    rvs1, ef1, m1, std1 = samples_from_gp(gpr,samples,new,observed,output,noise)
    gpr.kernel_params = kp[1]
    rvs2, ef2, m2, std2 = samples_from_gp(gpr,samples,new,observed,output,noise)
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(observed,output,"+")
    plot_samples(kp[0],rvs1,ef1,new,m1,std1)
    plt.subplot(2,1,2)
    plt.plot(observed,output,"+")
    plot_samples(kp[1],rvs2,ef2,new,m2,std2)
    for i in range(len(bounds)):
        plt.figure(2 + i)
        bk = list(bounds[i].keys())
        bv = list(bounds[i].values())
        gpr.ll_plot(observed,output,noise, keys = bk, bounds=bv ,n = 40)  
    plt.show()



def log_likelihood(gpr,new, observed, output,noise,kp):
    x =1 

def samples_from_gp(gpr,samples,new,observed,output,noise): 
    m,var = gpr.predict(new,observed, output, noise)
    rvs = multivariate_normal(mean = m,cov =var,allow_singular = True ).rvs(samples).T
    ef = np.sum(rvs,axis = 1)/samples
    std =  np.diag(var)
    return rvs,ef,m,std


def plot_samples(kp,rvs,ef,new,m,std):
    plt.plot(new, ef,"red")
    plt.plot(new, rvs[:,0],"gray", linestyle = "dotted")
    plt.fill_between(new, m - 2*std, m + 2*std, color= "lightblue")
    plt.title("params = "+ str(kp))
