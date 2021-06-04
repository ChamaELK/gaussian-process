import psdata
from scipy.signal import welch

t, x = psdata.cumsum(3000,4500)

dt = t[1:]- t[:-1]
fs = 1/np.mean(dt)
f, Pxx_den = signal.welch(x, fs)

plt.semilogy(f, Pxx_den)

plt.ylim([0.5e-3, 1])

plt.xlabel('frequency [Hz]')

plt.ylabel('PSD [V**2/Hz]')

plt.show()
