
import math, numpy as np, itertools, collections, scipy.special as special, json, os
p=0.5
r=3
pk=[special.comb(k+r-1,k)*(p**r)*((1-p)**k) for k in range(30)]
prob10=pk[10]
mean=sum(k*pk[k] for k in range(30))
second=sum(k*k*pk[k] for k in range(30))
print('prob10',prob10, 'mean', mean, 'second', second)
