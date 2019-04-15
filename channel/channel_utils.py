##############################################################################
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
##############################################################################
##############################################################################

##############################################################################
#
#                          __________        _______________
#Transmitted Signal ----> |Multipath| ----> |Changing Gains| ---->
#     ---> |+ Signals from Other Users| ---> |+ Broadband Noise| --->
#     ---> |+ Narrowband Noise| ---> Received Signal
#
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
    return noise

###############################################################################
#LPF Filter() : Generates Low Pass Filter FIR
###############################################################################
def window_lpf_fir(wc=0.5*np.pi, window='Cos', numTaps=255):
    filt = (wc/np.pi) * np.sinc((wc/np.pi)*np.linspace(-(numTaps-1)/2, (numTaps-1)/2, numTaps))
    if window == 'Cos':
        filt_win = np.cos(np.linspace(0, np.pi/2, 16))**2
        firFilt = np.concatenate((filt[:16]*np.flip(filt_win), filt[16:-16], filt[-16:]*filt_win))
    else:
        firFilt = filt
    return firFilt

###############################################################################
#BPF Filter() : Generates Band Pass Filter FIR
###############################################################################
def window_bpf_fir(w1=0.25*np.pi, w2=0.5*np.pi, window='Cos', numTaps=255):
    filt = (w2/np.pi) * np.sinc((w2/np.pi) * np.linspace(-(numTaps-1)/2, (numTaps-1)/2, numTaps)) - (w1/np.pi) * np.sinc((w1/np.pi) * np.linspace(-(numTaps-1)/2, (numTaps-1)/2, numTaps))
    if window == 'Cos':
        filt_win = np.cos(np.linspace(0, np.pi/2, 16))**2
        firFilt = np.concatenate((filt[:16]*np.flip(filt_win), filt[16:-16], filt[-16:]*filt_win))
    else:
        firFilt = filt
    return firFilt

###############################################################################
# Delta Function() : Generates Dirac Delta func of length "length"
#                    x[10] = 1, by default
###############################################################################
def delta_func(pos=10,length=4096):
    Ts = 1/(12*20*15000)
    t = np.linspace(0, Ts, num=length, endpoint=False)
    delta = np.concatenate((np.zeros(pos), np.ones(1), np.zeros(length-(pos+1))))
    return delta

###############################################################################
# Impulse Train Function() : Generates impulse train of length "length"
#                                gap = 0, length=25, by default
#                                gap defines intervals of 0s in between 1s
###############################################################################
def pulse_train(gap=0, length=25):
    Ts = 1/(12*20*15000)
    t = np.linspace(0, Ts, length, endpoint=False)
    x = np.zeros(length)
    for i in range(length):
        if(i%(gap+1)==0):
            x[i] = 1
    return x
