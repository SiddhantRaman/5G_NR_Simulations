##############################################################################
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
##############################################################################

###############################################################################
#AWGN_NOISE() : Generates Additive White Gaussian Noise of PSD power in dBm/Hz
#    AWGN has Gaussian PDF with 0 mean and Sigma^2 = Noise Power(No/2)
#    Noise dBm/Hz = 10Log(No/2.BW)
#    sigma = sqrt(No/2) = sqrt(BW*10^(Noise dBm/Hz/10))
###############################################################################
def awgn_noise(length, power, Bandwidth):
    sigma = np.sqrt(Bandwidth * 10**(power/10))
    noise = np.random.normal(0, sigma, length) # Manually set value of Variance to get desired PSD of AWGN Noise
    #fig1 = plt.figure()
    #plt.hist(noise)
    #fig2 = plt.figure()
    #plt.psd(noise, 4096, 3.6*10**6)
    #plt.show()
    return noise

def window_lpf_fir(wc=0.5*np.pi, window='Cos', numTaps=255):
    filt = (wc/np.pi) * np.sinc((wc/np.pi)*np.linspace(-(numTaps-1)/2, (numTaps-1)/2, numTaps))
    if window == 'Cos':
        filt_win = np.cos(np.linspace(0, np.pi/2, 16))**2
        firFilt = np.concatenate((filt[:16]*np.flip(filt_win), filt[16:-16], filt[-16:]*filt_win))
    else:
        firFilt = filt
    return firFilt

def window_bpf_fir(w1=0.25*np.pi, w2=0.5*np.pi, window='Cos', numTaps=255):
    filt = (w2/np.pi) * np.sinc((w2/np.pi) * np.linspace(-(numTaps-1)/2, (numTaps-1)/2, numTaps)) - (w1/np.pi) * np.sinc((w1/np.pi) * np.linspace(-(numTaps-1)/2, (numTaps-1)/2, numTaps))
    if window == 'Cos':
        filt_win = np.cos(np.linspace(0, np.pi/2, 16))**2
        firFilt = np.concatenate((filt[:16]*np.flip(filt_win), filt[16:-16], filt[-16:]*filt_win))
    else:
        firFilt = filt
    return firFilt

def delta_func(length=4096):
    Ts = 1/(12*20*15000)
    t = np.linspace(0, Ts, length, endpoint=False)
    delta = np.concatenate((np.zeros(10), np.ones(1), np.zeros(length-11)))
    return delta

# Generate AWGN noise of Power -130dBm/Hz and BW 3.6MHz
noise = awgn_noise(300000, -130, 12*20*15000)
fig1 = plt.figure()

# Plot Noise PSD in Blue
plt.psd(noise, 4096, 12*20*15000)

# Generate 255 Tap FIR BPF from pi/4 till 3pi/4 with rectangular window in Red
firFilt = window_bpf_fir(0.25*np.pi, 0.75*np.pi, 'None', 255)
# Filter the Noise with this filter
filtered_noise = np.convolve(noise, firFilt, mode='same')
plt.psd(filtered_noise, 4096, 3.6*10**6)

# Generate 255 Taps FIR BPF from pi/4 till 3pi/4 with Cos^2 Window in Green
firFilt = window_bpf_fir(0.25*np.pi, 0.75*np.pi, 'Cos', 255)
# Filter the Noise with this filter
filtered_noise = np.convolve(noise, firFilt, mode='same')
plt.psd(filtered_noise, 4096, 3.6*10**6)

# Plot the Un-Filtered Noise and Filtered Noise
plt.title('FIR Filter : Rect Window Vs Cos^2 Window')

# Plot the spectrum of a delta function : 
#    0dBm/Hz constant for all frequencies
fig2 = plt.figure()
d = delta_func(4096)
print(d.size, d[:10])
plt.plot(np.linspace(0, 4095, 4096), 20*np.log10(np.abs(np.fft.fft(d))) ,'-r')
plt.plot(np.linspace(0, 4095, 4096), np.imag(np.fft.fft(d))/np.real(np.fft.fft(d)), '-g')
plt.show()
