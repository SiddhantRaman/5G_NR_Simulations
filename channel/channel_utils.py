##############################################################################
# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def nTapChannel(numTaps=10, Ts=1/100):
    length = numTaps
    a = np.rand(numTaps)
