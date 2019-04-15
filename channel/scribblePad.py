##############################################################################
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import channel_utils as cu
##############################################################################

###############################################################################
############## SCRIPTS #############################///########################
###############################################################################

# Generate AWGN noise of Power -130dBm/Hz and BW 3.6MHz
noise = cu.awgn_noise(300000, -130, 12*20*15000)
fig1 = plt.figure()

# Plot Noise PSD in Blue
plt.psd(noise, 4096, 12*20*15000)

# Generate 255 Tap FIR BPF from pi/4 till 3pi/4 with rectangular window in Red
firFilt = cu.window_bpf_fir(0.25*np.pi, 0.75*np.pi, 'None', 255)
# Filter the Noise with this filter
filtered_noise = np.convolve(noise, firFilt, mode='same')
plt.psd(filtered_noise, 4096, 3.6*10**6)

# Generate 255 Taps FIR BPF from pi/4 till 3pi/4 with Cos^2 Window in Green
firFilt = cu.window_bpf_fir(0.25*np.pi, 0.75*np.pi, 'Cos', 255)
# Filter the Noise with this filter
filtered_noise = np.convolve(noise, firFilt, mode='same')
plt.psd(filtered_noise, 4096, 3.6*10**6)

# Plot the Un-Filtered Noise and Filtered Noise
plt.title('FIR Filter : Rect Window Vs Cos^2 Window')

# Plot the spectrum of a delta function :
#    0dBm/Hz constant for all frequencies
#fig2 = plt.figure()
d = cu.delta_func(100,4096)
#print(d.size, d[:110])
#plt.plot(np.linspace(0, 4095, 4096), 20*np.log10(np.abs(np.fft.fft(d))) ,'-r')
#plt.plot(np.linspace(0, 4095, 4096), np.imag(np.fft.fft(d))/np.real(np.fft.fft(d)), '-g')

#fig3 = plt.figure()
length = 100
train1 = cu.pulse_train(20, length)
train2 = cu.pulse_train(25, length)
#plt.stem(np.linspace(0, length-1, length), np.abs(np.fft.fft(train1)), '-b')
#plt.stem(np.linspace(0, length-1, length), np.abs(np.fft.fft(train2)), '-r')

#fig4 = plt.figure()
#plt.stem(np.linspace(0, length-1, length), train1, '-b')
#plt.stem(np.linspace(0, length-1, length), train2, '-r')

############## Convolve example #########################
Ts = 0.01
time = 10
length = int(time/Ts)
t = np.linspace(0, time, num=length, endpoint=False)
h = np.exp(-t)
x = np.zeros(length)
x[int(1/Ts)] = 3
x[int(3/Ts)] = 2
y = np.convolve(h, x, 'full')
fig5 = plt.figure()
plt.stem(t, y[:length], '-c')
plt.stem(t, x, '-r')
plt.stem(t, h, '-g')
plt.show()
