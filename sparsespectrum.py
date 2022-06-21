import phdcsv
import sys
import time as time_m
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch, detrend, butter,filtfilt, normalize
import math
from scipy.linalg import cholesky, cho_solve, det
from scipy.optimize import minimize
import pandas as pd
from numpy.linalg import norm
from scipy.stats import variation
from scipy import signal

#colinder 
guide_file = "logs/cvastrophoto_guidelog_20220523T051128.txt"
pulse_file = "logs/cvastrophoto_pulselog_20220523T051128.txt"
n= 7695
header=10
"""
Lambda centauri:

guide_file = "logs/cvastrophoto_guidelog_20210612T202456.txt"       
pulse_file = "logs/cvastrophoto_pulselog_20210612T202456.txt"
n= 6000
header= 10
"""
"""
pulse_file = "logs/cvastrophoto_pulselog_20210703T014040.txt"
guide_file = "logs/cvastrophoto_guidelog_20210703T014040.txt"
n=8579 
header = 10
"""
"""
# ioptron cem40
pulse_file = "logs/cvastrophoto_pulselog_20210801T170519.txt"
guide_file = "logs/cvastrophoto_guidelog_20210801T170519.txt"
n = 5749 
header =10
"""
"""
# python3 sparsespectrum.py 1200 0 
guide_file = "logs/cvastrophoto_guidelog_20201030T043254.txt"
pulse_file = "logs/cvastrophoto_pulselog_20201030T043254.txt"
n = 3000
header = 10
"""
guide_time, dither,  ur, ud, xr, xd = phdcsv.ra_dec_data(n,guide_file,plot=False)
pulse_time, pr,pd = phdcsv.pulse_log(pulse_file,header)

lower_bound = 3100
upper_bound = 4100

time = guide_time[lower_bound:upper_bound]
tresh = 0.6

_xr = np.zeros(upper_bound-lower_bound)

for i in range(upper_bound-lower_bound):
  if dither[lower_bound +i] == 0 and abs(xr[lower_bound +i]) < tresh:
    _xr[i] = xr[lower_bound +i]
  if dither[lower_bound +i] == 1:
    _xr[i] = np.nan


pointing_err_ra = np.nan_to_num(_xr)
pulse_lower_bound_0 = np.max(np.where(pulse_time<= guide_time[lower_bound+1]))
current_lower_bound = pulse_lower_bound_0
cumsum_pulse= []

for i in range(upper_bound -lower_bound-1):
  _next =0
  while pulse_time[current_lower_bound +_next] <= guide_time[i+1] :
    _next += 1
  cumsum_pulse.append(sum(pr[pulse_lower_bound_0 : (current_lower_bound +_next)]) + pointing_err_ra[i+1])
  current_lower_bound += _next

accumulated_gear_error_ra = np.array(cumsum_pulse)
#plt.plot(accumulated_gear_error_ra)
#plt.plot(pulse_ra)
#plt.plot(500*dither)
#plt.show()
#plt.savefig("fft_denoise/cumul.jpeg")
#plt.clf()
# power spectrum
post_dc= 5
ps_err = np.fft.fft(accumulated_gear_error_ra)**2
freq = np.fft.fftfreq(n,min(time))
half_n = n//2
ps_err_half = (2.0 / n) * ps_err[post_dc:half_n]
freq_half = freq[post_dc:half_n]

#plt.plot(freq_half, np.abs(ps_err_half))
#plt.xlabel("Frequency (Hz)")
#plt.ylabel("Power spectrum")
#plt.savefig("fft_denoise/ps.jpeg")
#plt.clf()

# spectral density using welch 
f , welch_err = welch(detrend(accumulated_gear_error_ra) , fs=1/min(time), window='hann', nperseg= 256, return_onesided=True)
plt.plot(f, welch_err)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectral density using Welch")
plt.savefig("fft_denoise/welch.jpeg")
plt.clf()
# butterworth filter for denoising
fs = 1/min(time)
nyq = fs* 0.5
cutoff = 0.8 * 10**-10
order = 2
normal_cutoff = cutoff / nyq
print(normal_cutoff)
# Get the filter coefficients 
b, a = butter(order, normal_cutoff, btype='low', analog=False)
y = filtfilt(b, a, detrend(accumulated_gear_error_ra))


plt.plot(detrend(accumulated_gear_error_ra))
plt.plot(y,color='red')
plt.savefig("fft_denoise/butterworth.jpeg")
plt.clf()

f_y , welch_y = welch(y , fs=1/min(time), window='hann', nperseg= 512, return_onesided=True)

#plt.plot(f_y, welch_y)
#plt.show()
#f_welch_err = np.vstack([f,welch_err])    
#sr0 = f_y[welch_y.argsort()[-4:]]
#sr0.sort()
#print(sr0)
#sr0 = f[:40]
#sr0 = np.array([0.18,0.24,0.5])
#m= len(sr0)

def acf(x):
    length = len(x)
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y



def trigonometric(x,sr):
    m = len(sr)
    trigonometric = np.zeros(2*m)
    for i in range(m):
        trigonometric[2*i] = math.cos(2*math.pi*sr[i]*x)
        trigonometric[2*i+1] = math.sin(2*math.pi*sr[i]*x)
    return trigonometric

def design_matrix(x,s):
    m = len(s)
    design_matrix = np.zeros(shape= (n,2*m))
    for i in range(n):
        for j in range(m) :
            design_matrix[i,2*j] =  math.cos(2*math.pi*s[j]*x[i])
            design_matrix[i,2*j+1] =  math.sin(2*math.pi*s[j]*x[i])
    return design_matrix

def matrixA(train_input , s , sig_n , sig_0):
    m = len(s)
    Gf = design_matrix(train_input,s)
    return Gf.T@Gf + (m * sig_n**2 / sig_0**2 * np.identity(2*m))

def gp_ll(train_input, train_output, parameters):
    sig_n = parameters[-2]
    sig_0 = parameters[-1]
    sr= parameters[:-2]
    m = len(sr)
    n = len(train_input)
    A = matrixA(train_input, sr, sig_n, sig_0)
    lower = True
    R = cholesky(A,lower=lower)
    B = cho_solve((R, lower) , design_matrix(train_input,sr).T @ train_output)
    return -0.5* (norm(train_output, ord=2)**2 - norm(B, ord=2)**2)/sig_n**2 - 0.5* np.sum(np.diag(R)**2) + m*math.log(m*sig_n**2/sig_0**2) - 0.5*n*math.log(2*math.pi*sig_n**2) 
    
def gp_prediction(prediction_input,train_input, train_output, s, sign_n, sig_0):
    m = len(s)
    n = len(train_input)
    G = trigonometric(prediction_input,s) 
    A = matrixA(train_input, s, sig_n, sig_0)
    lower = True
    R = cholesky(A,lower=lower)
    B = cho_solve((R,lower), design_matrix(train_input,s).T @ train_output)
    gp_mean =  sig_0**2/m * G.T @ cho_solve((R.T, lower), B)
    C = cho_solve((R.T,lower), G)
    gp_var =  math.sqrt(sig_n**2 * ( 1 + norm(C,ord=2)**2))
    return gp_mean, gp_var

def reporter(p):
    """Reporter function to capture intermediate states of optimization."""
    global p_sig_n
    global p_sig_0
    p_sig_n.append(p[-2])
    p_sig_0.append(p[-1])

def variance_signal(data):
    return math.sqrt(np.var(butter_highpass_filter(data, 2.4e-10,1/min(time)), ddof=1 ))

# RADriftSpeed	DECDriftSpeed

n = 200
data = pointing_err_ra
NP = 200
m_prediction = np.zeros(NP)
v_prediction =np.zeros(NP)
prediction_inputs= np.zeros(NP)
frequency_training = 20
for i in range(NP):
    train_output = data[-n-NP+i-1:-NP+i-1]
    train_input = time[-n-NP+i-1:-NP+i-1]
    prediction_inputs[i] = time[-NP+i]
    pf ,pwelch = welch(acf(data[-n-NP+i:-NP+i]) , fs=2, window='hann', nperseg= 256, return_onesided=True)
    sig_0 = np.sum(pf**2)
    sr0 = pf[pwelch.argsort()[-20:]]
    sr0.sort()
    m= len(sr0)
    if i%frequency_training == 0:
        sig_n =  math.sqrt(np.var(butter_highpass_filter(train_output[-50:], 0.4e-10,1/min(time)), ddof=1 ))
        #prediction_input = time[n+i+1]
        parameters = np.append(sr0,[sig_n,sig_0])
        f= lambda params : gp_ll(train_input, train_output, params)
        result = minimize(f,parameters,options={"maxiter":1}, method="L-BFGS-B")
        parameters = result.x
    p_sig_n = [sig_n]
    p_sig_0 = [sig_0]
    logl = result.fun
    sig_n = parameters[-2]
    sig_0 = parameters[-1]
    sr_min = parameters[:-2]
    m_prediction[i], v_prediction[i] = gp_prediction(prediction_inputs[i], train_input, train_output, sr_min, sig_n, sig_0)
    print(sig_n,sig_0,sr_min,m_prediction[i],v_prediction[i])

uncertainty = 0.5* np.sqrt(v_prediction)
plt.subplot(2,1,1)
plt.scatter(time[0:n+NP],data[0:n+NP])
plt.subplot(2,1,2)
plt.plot(prediction_inputs, m_prediction,color='green')
#for i in range(NP):
plt.fill_between(prediction_inputs,m_prediction+uncertainty, m_prediction- uncertainty, alpha=0.1)
plt.show()

#plt.savefig('fft_denoise/prediction.png')
#plt.clf()



