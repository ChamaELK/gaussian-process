import gpkplots
import numpy as np

def simplekernels():
    samples = 5
    m = 101
    new = np.linspace(-5,5,m)
    observed = np.array([-4,-3,-1,0,2])
    output = np.array([-2,0,1,2,-1])
    noise = 0.05 
    gpkplots.figures_kernels(new, observed, output, noise)

simplekernels()
