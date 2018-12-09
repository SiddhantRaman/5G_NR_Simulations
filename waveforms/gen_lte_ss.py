# Generate Primary and Secondary Synchronization Sequence
# This file will also contain generation of full frame of OFDM symbols
# PSS and SSS
# Later in the file we will be performing the synchronization for lte

import numpy as np
import matplotlib.pyplot as plt

# Define Global Constants
SQRT_2 = np.sqrt(2)
PI     = np.pi
ROOT = np.array([25, 29, 34])
# Primary Synchrnization sequence of LTE is based on Zadoff-chu Sequence of
# length 62

# Generate Primary Synchronization sequence
# PSS is mapped to 72 Sub-Carriers(6 RBs), centered around DC
# Length 62 Zadoff-chu Sequence
def gen_lte_pss(N_ID_2):
    if(N_ID_2 >2 or N_ID_2 <0):
        print("ERROR: N_ID_2 can take only values from {0, 1, 2}")
        return -1
    root_ind = ROOT[N_ID_2]
    print(root_ind)

    n = np.linspace(0,30,31)
    pss = np.zeros(62) + 1j*np.zeros(62) #including DC
    pss[0:31] = np.exp(-1j*PI/63*root_ind*n*(n + np.ones(31)))

    n = np.linspace(31,61,31)
    pss[31:62] = np.exp(-1j*PI/63*root_ind*(n + np.ones(31))*(n + 2*np.ones(31)))

    return pss

# Generate Secondary Synchronization sequence
# SSS is mapped to 72 Sub-carriers(6 RBs), centered around DC
# SSS in Slot0(SubFrame0) is different that SSS in Slot10(SubFrame5)
# Length 62
def gen_lte_sss(N_ID_1, slotNum):
    
pss = gen_lte_pss(1)
print(pss)
plt.scatter(np.real(pss), np.imag(pss))
plt.show()
