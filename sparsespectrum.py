import phdcsv
import sys
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
lower_bound = int(sys.argv[1])
"""
Lambda centauri:

guide_file = "logs/cvastrophoto_guidelog_20210612T202456.txt"       
pulse_file = "logs/cvastrophoto_pulselog_20210612T202456.txt"
n= 6000
header= 10
"""
pulse_file = "logs/cvastrophoto_pulselog_20210703T014040.txt"
guide_file = "logs/cvastrophoto_guidelog_20210703T014040.txt"
n=8579 
header = 10


"""
# python3 sparsespectrum.py 1200 0 
guide_file = "logs/cvastrophoto_guidelog_20201030T043254.txt"
pulse_file = "logs/cvastrophoto_pulselog_20201030T043254.txt"
n = 3000
header = 10
"""
index ,data = phdcsv.pulse_guide(guide_file, pulse_file,n, header)
total = len(data)
value = int(sys.argv[2])
upper_bound = total if value == 0 else value
time = data[lower_bound:upper_bound,index['time']]

dither = data[lower_bound:upper_bound, index['dither']]

#right_assention 
pulse_ra = data[lower_bound:upper_bound,index['pr']] 
pulse_ra *= 1- dither
pointing_err_ra = (1-dither)*data[lower_bound:upper_bound, index['xr']] * 1000
# accumulated gear error at time i 
# measure pointning error at time i + the cumulative sum of the controling error from 0 to i
n = len(pulse_ra)
accumulated_gear_error_ra = np.zeros(n, dtype= np.float)

for i in range(n):
    cumsum = 0 
    for j in range(i):
            cumsum += pulse_ra[j]
            accumulated_gear_error_ra[i] =  pointing_err_ra[i] + cumsum



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

plt.plot(freq_half, np.abs(ps_err_half))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power spectrum")
plt.savefig("fft_denoise/ps.jpeg")
plt.clf()

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


pf ,pwelch = welch(acf(detrend(accumulated_gear_error_ra)) , fs=1/min(time), window='hann', nperseg= 256, return_onesided=True)
sr0 = pf[pwelch.argsort()[-15:]]
sr0.sort()
m= len(sr0)
n = 1200
data = detrend(accumulated_gear_error_ra)
NP=300
prediction_inputs = time[n+1:n+NP+1]
m_prediction = np.zeros(NP)
v_prediction =np.zeros(NP)


sig_0 = 35
first = True
for i in range(NP):
    train_output = data[i:n+i]
    train_input = time[i:n+i]
    sig_n =  math.sqrt(np.var(butter_highpass_filter(train_output, 2.4e-10,1/min(time)), ddof=1 ))
    #prediction_input = time[n+i+1]
    if first : 
        f= lambda params : gp_ll(train_input, train_output, params)
        start = np.append(sr0,[sig_n,sig_0])
        parameters= start
        p_sig_n = [sig_n]
        p_sig_0 = [sig_0]
        result = minimize(f,start,options= {"maxiter":50},callback=reporter, method="L-BFGS-B")
        parameters = result.x
        logl = result.fun
        sig_n = parameters[-2]
        sig_0 = parameters[-1]
        sr_min = parameters[:-2]
    first = False
    m_prediction[i], v_prediction[i] = gp_prediction(prediction_inputs[i], train_input, train_output, sr_min, sig_n, sig_0)

uncertainty = v_prediction

plt.plot(time[0:n+NP],data[0:n+NP])
plt.plot(prediction_inputs, m_prediction,color='green')
#for i in range(NP):
plt.fill_between(prediction_inputs,m_prediction+uncertainty, m_prediction- uncertainty)
plt.show()

#plt.savefig('fft_denoise/prediction.png')
#plt.clf()



