# downlink_waveform.py

############################################################################
# Functions defined here will be used to generate downlink OFDM 5G NR waves
############################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def cp_ofdm_symbol(data, mu=0, Nrb=20, Guard=7):
    deltaF = 2^mu * 15000
    NoOfCarriers = 12*Nrb #12 Sub-Carriers per RB(Resource Block)
    NFFT = 4096

    Bandwidth = deltaF * NoOfCarriers

    FftData = np.concatenate((np.zeros(1), data[:int((NoOfCarriers-2*Guard-1-1)/2)], np.zeros(2*Guard)))
    FftData = np.concatenate((FftData, data[int((NoOfCarriers-2*Guard-1-1)/2):], np.zeros(NFFT-NoOfCarriers)))

    ofdm_symbol = np.fft.ifft(FftData, NFFT)
    #print(ofdm_symbol.size)
    #fig1 = plt.figure()
    #plt.plot(np.linspace(0, 2*np.pi, NoOfCarriers), 20*np.log10(np.fft.fft(ofdm_symbol)[:NoOfCarriers]), '-g')
    #fig2 = plt.figure()
    #plt.stem(np.linspace(0, NFFT, NFFT), ofdm_symbol, '-g')

    #fig3 = plt.figure(figsize=(100, 5))
    #plt.stem(np.linspace(0, NFFT, NFFT), FftData, '-r')

    #plt.show()
    return ofdm_symbol

def cp_ofdm_symbol_cp(data, mu=0, Nrb=20, Guard=7, CPSize=288):
    TimeDomainSymbol = cp_ofdm_symbol(data, mu, Nrb, Guard)
    NoOfCarriers = Nrb*12

    TimeDomainSymbolCP = np.concatenate((TimeDomainSymbol[-CPSize:], TimeDomainSymbol))
    x = 113*np.pi/60 * np.sinc(113/120 * np.linspace(-255, 255, 511))
    x_win = x * (0.5*(1 + np.cos(2*np.pi/510 * np.linspace(-255, 255, 511))))**0.6
    TxSymbolFilt = signal.convolve(TimeDomainSymbolCP, x_win, mode='same')/sum(x_win)
    return TxSymbolFilt

def cp_ofdm_waveform(mu=0, Nrb=20, Guard=7):
    NoOfCarriers = 12*Nrb
    TxSymbolOut = np.zeros(4096)
    for i in np.linspace(0, 2000, 2000):
        data = np.random.randint(2, size=20*12-15)
        data[data == 0] = -1
        TxSymbolOut = TxSymbolOut + np.abs(np.fft.fft(cp_ofdm_symbol_cp(data, 0, 20, 7, 288)[288:]))
    TxSymbolOut = TxSymbolOut/2000
    fig1 = plt.figure()
    plt.plot(np.linspace(0, 2*np.pi, NoOfCarriers), 20*np.log10(TxSymbolOut[:NoOfCarriers]))
    plt.show()
    print(0) 

def ofdm_dl_waveform_5g(mu=0, Nrb=20, Guard=7):
    #########################################
    NFFT = 4096
    NoOfCarriers = 12*Nrb
    deltaF = 2**mu * 15000
    CPSize = 288
    CPSize_FirstSymbol = 320
    subFrameDur = 1/1000 # 1msec
    FrameDur = 10*subFrameDur #10 msec
    T_symbol = 1/deltaF
    deltaF_max = deltaF * 2**5 # 480KHz
    deltaF_ref = deltaF
    NFFT_ref = 2048
    Bandwidth = deltaF * NoOfCarriers
    ##########################################
    #Derived Params
    kappa = deltaF_max*NFFT/deltaF_ref/NFFT_ref
    Nu = NFFT_ref * kappa /2**mu #Num of OFDM samples in 32 symbols
    Ncpmu = 144*kappa/2**mu # total CP samples in 32 symbols
    Ts = 1/deltaF_max/NFFT # symbolDur if Symbol occupies 480*4096 KHz (Max bandwidth possible in 5G NR with 4096 tones)
    ###########################################

    t = np.linspace(0, (Nu+Ncpmu)*Ts, NFFT + CPSize)
    data = np.loadtxt('modData.txt')
    NumData = NoOfCarriers - 2*Guard - 1
    a_k = np.concatenate((np.zeros(Guard), data[:int((NumData+1)/2)], np.zeros(1)))
    a_k = np.concatenate((a_k, data[int((NumData+1)/2):int(NumData)], np.zeros(Guard)))
    k = np.linspace(- int(NoOfCarriers/2), int(NoOfCarriers/2)-1, NoOfCarriers)
    print(t.size)
    S_t = np.zeros(t.size)
    for i in k:
        S_t = a_k[int(i)] * np.exp(1j*2*np.pi*i*deltaF*(t - Ncpmu*Ts)) + S_t

    fig1 = plt.figure()
    plt.stem(t, S_t, '-g')

    fig2 = plt.figure()
    CPdur = CPSize/NFFT * 1/deltaF
    Fs = (NFFT+CPSize)/(T_symbol + CPdur)
    print(Fs)
    f, Pxx_den = signal.periodogram(S_t, Fs)
    plt.semilogy(f, Pxx_den, '-g')
    f, Pxx_den = signal.welch(S_t, Fs)
    plt.semilogy(f, Pxx_den, '-r')
    plt.xlim((-(Bandwidth+2*10**6)/2, (Bandwidth+2 * 10**6)/2))
    plt.ylim((10**-10, 2*np.max(Pxx_den)))
    plt.show()

ofdm_dl_waveform_5g(0, 20, 7)
