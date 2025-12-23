
import math, scipy.special as sp
r=3
p=0.5
k=10
prob10=sp.comb(k+r-1,k)*(p**r)*((1-p)**k)
prob10