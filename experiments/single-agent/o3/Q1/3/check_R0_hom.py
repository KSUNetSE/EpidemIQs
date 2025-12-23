
import numpy as np, pandas as pd, os, math, json
# Parameters for analytical ODE integration
beta_h=0.5  # choose baseline values to yield R0 2.5 given gamma=0.2 and mean degree 10? Not critical.
sigma=1/3
gamma=1/5
N=1.0
# compute R0 in homogenous mixing: beta_h/gamma*1 (assuming S0~1)
R0=beta_h/gamma
R0
