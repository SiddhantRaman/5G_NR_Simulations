# Generate DL Reference Sequences
import numpy as np
import matplotlib.pyplot as plt
# Generate Cell Specific Reference Sequence
# rl,ns(m) = 1/sqrt(2).(1 - 2.c(2m)) +j1/sqrt(2).(1 - 2.c(2m+1)), m = 0,1,2,3,...,2Nrb, max - 1
# c(n) = (x(n+Nc) + y(n+Nc))mod2
# x(n+31) = (x(n+3) + x(n))mod 2
# y(n+31) = (y(n+3) + y(n+2) + y(n+1) + y(n))mod 2
# c_init and Length of the psedo-random sequence (c(n)) needs to be specified
# inputs : Ns : Slot number within a Frame
#          L  : Lth OFDM Symbol within the slot
#          N_CellID : Physical Layer Cell ID
# output : crs_lte : Cell Specific Reference Sequence

SQRT_2 = np.sqrt(2)
PI     = np.pi

def gen_lte_crs(Ns, L, N_CellID):
    N_cp = 1
    N_rb_dlmax = 110
    c_init = 1024*(7*(Ns +1) + L + 1)*(2*N_CellID + 1) + 2*N_CellID + N_cp
    # Generate Gold Sequence of len = 2*N_rb_dlmax
    length = 2 * N_rb_dlmax
    x = np.zeros(1600 + length)
    y = np.zeros(1600 + length)
    c_init_tmp = c_init
    for n in range(0,31):
        y[30-n+1] = np.floor(c_init_tmp/np.power(2,(30-n)))
        c_init_tmp = c_init_tmp - np.floor(c_init_tmp/np.power(2,(30-n))) * np.power(2,(30-n))
    x[1] = 1

    for n in range(0, 1600+length-32):
        x[n+31+1] = np.mod((x[n+3+1] + x[n+1]), 2)
        y[n+31+1] = np.mod((y[n+3+1] + y[n+2+1] + y[n+1+1] + y[n+1]), 2)
    c = np.zeros(length)
    for n in range(1, length):
        c[n] = np.mod((x[n+1600] + y[n+1600]), 2)

    crs = np.zeros(N_rb_dlmax) + 1j*np.zeros(N_rb_dlmax)
    for n in range(0, N_rb_dlmax):
        crs[n] = 1/SQRT_2 * (1 - 2*c[2*n]) + 1j*(1/SQRT_2 * ( 1 - 2*c[2*n + 1]))
    return crs

crs = gen_lte_crs(4, 4, 10)
print(len(crs))
print(crs)
plt.scatter(np.real(crs), np.imag(crs))
plt.show()
