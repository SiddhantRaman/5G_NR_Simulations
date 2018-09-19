# bitStreampy.py

###################################################################
# Generates different bitStreams to be used for OFDM symbol data
###################################################################
import numpy as np

def random_bit_stream(length=1000000, modulation='BPSK'):
    if modulation == 'BPSK':
        bitStream = np.random.randint(2, size=length)
        bitStream[bitStream == 0] = -1
    else:
        bitStreamI = np.random.randint(2, size=length)
        bitStreamI[bitStreamI == 0] = -1
        bitStreamQ = np.random.randint(2, size=length)
        bitStreamQ[bitStreamQ == 0] = -1
        bitStream = bitStreamI + 1j* bitStreamQ
    np.savetxt('modData.txt', bitStream, fmt='%d')

random_bit_stream(1000000, 'BPSK')

