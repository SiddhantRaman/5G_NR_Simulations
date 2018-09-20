# downlink_waveform.py

############################################################################
# Functions defined here will be used to generate downlink OFDM 5G NR waves
############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 

def get_papr(Signal):
    length = Signal.size
    peakPwr = (np.abs(np.max(Signal)))**2
    avgPwr = np.abs(np.inner(Signal, Signal))/length
    return (10*np.log10(peakPwr/avgPwr))

def overlap_add(S, S_t, WinSize):
    WinSize = int(WinSize)
    if(WinSize > 0):
        retval = np.concatenate((S[:-WinSize], S[-WinSize:] + S_t[:WinSize], S_t[WinSize:]))
    else:
        retval = np.concatenate((S, S_t))
    return retval

########################################################################################################################
# Function to generate NumOfSymbols OFDM symbol based on waveform specified in 5G
# input params : NumOfSymbols, numerology(mu), Nrb (Resource Blocks), Guard tones on both side of spectrum
# output : NumOfSymbols 5G OFDM symbols with Cyclic Prefix of 288 samples
# Algorithm is simple to understand from the code
# I have used Cosine Windowing for generating WOLA OFDM symbols
########################################################################################################################
def ofdm_ul_waveform_5g(NumOfSymbols, mu, Nrb, Guard, WinSize):
    ####################################################################################################################
    NFFT = 4096                       # NFFT fixed in 5G 
    NoOfCarriers = 12*Nrb             # Num Carriers/Tones = 12 * Num Res Blocks
    deltaF = 2**mu * 15000            # sub-carrier spacing 2^mu * 15KHz
    CPSize = 288                      # Cyclic Prefix is 288
    CPSize_FirstSymbol = 320          # First Symbol of half SubFrame Cyc Prefix is 320
    subFrameDur = 1/1000              # 1msec
    FrameDur = 10*subFrameDur         # 10 msec
    T_symbol = 1/deltaF               # OFDM Symbol Duration without CP 
    deltaF_max = 15000 * 2**5         # 480KHz
    deltaF_ref = 15000                # LTE sub-carrier spacing fixed to 15KHz
    NFFT_ref = 2048                   # NFFT fixed in LTE

    ####################################################################################################################
    #Derived Params
    Bandwidth = deltaF * NoOfCarriers               # Sub-Carrier spacing * num Sub-Carriers
    kappa = (deltaF_max*NFFT)/(deltaF_ref*NFFT_ref) # Constant : max Number 5G symbols possible in LTE Symbol Duration
    Nu = NFFT_ref * kappa /(2**mu)                  # Num of OFDM samples in 32 symbols
    Ncpmu = 144*kappa/(2**mu)                       # total CP samples in 32 symbols
    Ts = 1/deltaF_max/NFFT # symbolDur if Symbol occupies 480*4096 KHz (Max bandwidth possible in 5G NR with 4096 tones)
    Fs = NFFT/T_symbol                              # Sampling Frequency to achieve desired Symbol Duration
    Fc = Bandwidth/2 - Guard*deltaF                 # Cut-off frequency for LPF FIR design
    ####################################################################################################################

    t = np.linspace(0, (Nu+Ncpmu)*Ts, NFFT + CPSize, endpoint=False) # NFFT+CP size time-domain sequence required 0<=t<(Nu+Ncpmu)Ts
    data = np.loadtxt('modData.txt')                                 # BPSK modulated random data stored in 'modData.txt'
    NumData = NoOfCarriers - 2*Guard - 1                             # Num of nPSK symbols which can be loaded on 1 OFDM Symbol
    # Window design for WOLA operation to suppress out of band spectral leakage
    if(WinSize>1):
        alpha = 0.5
        nFilt = np.linspace(-np.pi/2, 0, WinSize)
        #x = 4*np.pi*Fc/Bandwidth * np.sinc(2*Fc/Bandwidth * nFilt)
        #x_win = x * (np.cos(np.pi*alpha*nFilt/T_symbol))/(1 - (4 *(alpha**2) *np.multiply(nFilt,nFilt))/(T_symbol**2) )
        # Below is a most simple Cos-Squared window which will be applied to the time domain OFDM Symbol
        x_win = np.multiply(np.cos(nFilt), np.cos(nFilt))
        print(x_win)
    # Below Loop will run for Number of OFDM symbols to be generated
    for num in range(0, NumOfSymbols):
        # a_k is the sequence of nPSK symbols which will be loaded on Sub-carriers(FFT Tones)
        a_k = np.concatenate((np.zeros(Guard), data[NumData*num : NumData*num + int((NumData+1)/2)], np.zeros(1)))
        a_k = np.concatenate((a_k, data[NumData*num + int((NumData+1)/2):int(NumData) + NumData*num], np.zeros(Guard)))

        # k is sub-carrier index starting from most negative tone (I am generating DC centered OFDM Spectrum)
        k = np.linspace(- int(NoOfCarriers/2), int(NoOfCarriers/2)-1, NoOfCarriers)
        # S_t is time-domain OFDM symbol with CP. Above loop generates OFDM Symbol with CP appended already :)
        S_t = np.zeros(t.size)
        for i in k:
            S_t = a_k[int(i+NoOfCarriers/2)] * np.exp(1j*2*np.pi*i*deltaF*(t - Ncpmu*Ts)) + S_t

        # Apply windowing (WOLA) only when window size is greater than 1
        if(WinSize>1):
            S_t = np.concatenate((S_t , S_t[CPSize:CPSize + WinSize]))
            #S_t = np.convolve(S_t, x_win, mode='same')/sum(x_win)
            S_t[:WinSize] = S_t[:WinSize] * x_win
            S_t[-WinSize:] = S_t[-WinSize:] * np.flip(x_win)
        if(num == 0):
            S = S_t
        else:
            S = overlap_add(S, S_t, WinSize)
            
    print(S.size, Nu, Ncpmu)
    print(1/deltaF,(Nu + Ncpmu)*Ts)
    # small verification of OFDM Receiver with  no noise
    # Just remove CP and take FFT of Tx Symbol to get the data back
    print(a_k[120:220])
    if(WinSize>1):
        x1 = np.fft.fft(S[-4096-WinSize:-WinSize], 4096)/4096
    else:
        x1 = np.fft.fft(S[-4096:], 4096)/4096
    print(np.real(x1[:100]))
    #fig1 = plt.figure()
    #plt.title('Power Spectral Density (OFDM)')
    #plt.psd(S, 4096, Fs)
    #plt.xlim((-Nrb*7.5*deltaF, Nrb*7.5*deltaF))
    #plt.ylim((-80, -35))
    #plt.show()
    return S

#####################################################################################################
# UpLink in 5G NR supports 2 types of Waveform : 
#    ofdm_ul_waveform_5g() implements general OFDM like DL Waveform
#    dft_ofdm_ul_waveform_5g() implements DFT-S-OFDM to minimize the PAPR for power limited purposes
#####################################################################################################
def dft_ofdm_ul_waveform_5g(NumSymbols, mu, Guard, WinSize):
    print('TODO Activity!!!')

# Lets check our 5G NR Downlink Wave for mu = 1, Nrb = 20, Guard Tones = 7 and Window of length = 128
mu = 1
S1 = ofdm_ul_waveform_5g(14, mu, 20, 7, 0)
S2 = ofdm_ul_waveform_5g(14, mu, 20, 7, 128)
print(get_papr(S2))
Fs = 4096 * 15000 * 2**mu
fig1 = plt.figure()
plt.title('UL-PSD Comparison OFDM Vs OFDM-WOLA')
plt.set_cmap('jet')
plt.psd(S1, 4096, Fs)
plt.psd(S2, 4096, Fs)
plt.xlim((-2.75*10**6*2**mu, 2.75*10**6*2**mu))
plt.show()
