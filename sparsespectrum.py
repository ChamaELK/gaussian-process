import phdcsv
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch


lower_bound = int(sys.argv[1])
"""
Lambda centauri:

guide_file = "logs/cvastrophoto_guidelog_20210612T202456.txt"       
pulse_file = "logs/cvastrophoto_pulselog_20210612T202456.txt"
n= 6000
header= 10
"""

guide_file = "logs/cvastrophoto_guidelog_20201030T043254.txt"
pulse_file = "logs/cvastrophoto_pulselog_20201030T043254.txt"
n = 3000
header = 10

index ,data = phdcsv.pulse_guide(guide_file, pulse_file,n, header)
total = len(data)
value = int(sys.argv[2])
upper_bound = total if value == 0 else value
time = data[lower_bound:upper_bound,index['time']]

dither = data[lower_bound:upper_bound, index['dither']]

#right_assention 
pulse_ra = data[lower_bound:upper_bound,index['pr']] 
pulse_ra *= 1- dither
pointing_err_ra = data[lower_bound:upper_bound, index['xr']] * 1000
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
plt.show()

fft_err = np.fft.fft(accumulated_gear_error_ra)
freq = np.fft.fftfreq(n,min(time))
half_n = n//2
fft_err_half = (2.0 / n) * fft_err[:half_n]
freq_half = freq[:half_n]

plt.plot(freq_half, np.abs(fft_err_half))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()




