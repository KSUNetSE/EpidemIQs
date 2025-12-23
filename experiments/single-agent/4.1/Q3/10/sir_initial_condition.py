
import numpy as np
import os
# SIR initial condition
N = 1000
I0 = 5
R0_ = 0
S0 = N - I0 - R0_
init_static = {'percentage': {'S': int(S0/N*100), 'I': int(I0/N*100), 'R': int(R0_/N*100)}}
init_static['counts'] = {'S': S0, 'I': I0, 'R': R0_}

# Return initial condition for simulation setup
init_static