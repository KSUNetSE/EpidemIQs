
import math, itertools, collections, os, json, numpy as np
from math import comb
# Negative Binomial r=3, p=0.5 (mean 3, variance 6)
r=3
p=0.5
max_k=50
pk=[comb(k+r-1,k)*(p**r)*((1-p)**k) for k in range(max_k)]
p_total=sum(pk)
print('sum',p_total)
# compute <k> and <k^2>
mean_k=sum(k*pk[k] for k in range(max_k))
second=sum(k*k*pk[k] for k in range(max_k))
print(mean_k,second)
q=(second-mean_k)/mean_k
print('q',q)
# remove all k=10 nodes
pk2=pk.copy()
pk2[10]=0
remaining_prob=sum(pk2)
mean_k2=sum(k*pk2[k] for k in range(max_k))/remaining_prob
second2=sum(k*k*pk2[k] for k in range(max_k))/remaining_prob
q2=(second2-mean_k2)/mean_k2
print('after removal mean',mean_k2,'q2',q2)
new_R0=q2 # since beta/gamma=1
print('R0 new',q2)
