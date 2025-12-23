
import numpy as np,math
P10=0.008056
mean=3
second=15

def ratio(y):
    denom=1-y*P10
    if denom<=0: return float('inf')
    mean_new=(mean - y*P10*10)/denom
    sec_new=(second - y*P10*100)/denom
    return sec_new/mean_new

ys=np.linspace(0,1,101)
list_vals=[(y,ratio(y)) for y in ys]
min_ratio=min(val for y,val in list_vals)
print(min_ratio)
for y,val in list_vals:
    if val<2:
        print('threshold',y)
        break

import math, numpy as np, scipy.stats, sys, json, os, itertools, random, pandas as pd, matplotlib.pyplot as plt
import math, numpy as np
z=3.0
q=4.0
# For Poisson mean z, <k>=z, <k^2>=z^2+z (variance z)
second=z*z+z
print('second moment',second)
P10=math.exp(-z)*z**10/math.factorial(10)
print('P10',P10)

def kappa_after(alpha):
    # fraction removed among k=10
    rem=alpha*P10
    remaining=1-rem
    # new mean
    mean_prime=(z - alpha*10*P10)/remaining
    second_prime=(second - alpha*100*P10)/remaining
    kappa=(second_prime-mean_prime)/mean_prime
    return kappa,remaining,mean_prime,second_prime

# find alpha threshold
for alpha in np.linspace(0,1,1001):
    kappa,_rem,_m,_s=kappa_after(alpha)[:4]
    if kappa<=1.0:
        print('alpha needed',alpha)
        break