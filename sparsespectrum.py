import phdcsv
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch


lower_bound = int(sys.argv[1])
upper_bound = int(sys.argv[2])
index ,data = phdcsv.pulse_guide()
time = data[lower_bound:upper_bound,0]
#right_assention 
pulse_ra = data[lower_bound:upper_bound,1]
pointing_err_ra = data[lower_bound:upper_bound, 3] * 1000
# accumulated gear error at time i 
# measure pointning error at time i + the cumulative sum of the controling error from 0 to i
n = len(pulse_ra)
accumulated_gear_error_ra = np.zeros(n, dtype= np.float)

for i in range(n):
    cumsum = 0 
    for j in range(i):
        cumsum += pulse_ra[j]
    accumulated_gear_error_ra[i] =  pointing_err_ra[i] + cumsum

plt.plot(accumulated_gear_error_ra)
plt.savefig('accumulated_gear_error')

frequency = 1. / np.mean(time)
# welch method for approximation of the power spectrum 
f, Pxx_den = welch(accumulated_gear_error_ra, frequency)
plt.ylim([1, 1e4])
plt.xlim([1,400])
plt.semilogy(f, Pxx_den)
plt.savefig('freq_accumulated_gear_error')




