
import math
lam=3.0
p = [math.exp(-lam)*lam**k/math.factorial(k) for k in range(0,30)]
P10 = p[10]
print('P10',P10)
orig_mean = sum(k*p[k] for k in range(len(p)))
orig_second = sum(k*k*p[k] for k in range(len(p)))
q = (orig_second - orig_mean)/orig_mean
print('orig q',q)
remaining_prob = 1-P10
mean_new = sum(k*p[k] for k in range(len(p)) if k!=10)/remaining_prob
second_new = sum(k*k*p[k] for k in range(len(p)) if k!=10)/remaining_prob
q_new = (second_new - mean_new)/mean_new
print('new mean',mean_new,'new q',q_new)

import scipy.special as sp
r=3
p=0.5
max_k=100
pk=[sp.comb(k+r-1,k)*(p**r)*((1-p)**k) for k in range(max_k)]
S=sum(pk)
for k in range(max_k):
    pk[k]/=S
mean=sum(k*pk[k] for k in range(max_k))
second=sum(k*k*pk[k] for k in range(max_k))
q=(second-mean)/mean
# Remove nodes degree 10 from network (but keep degrees of others unchanged for this approximate analysis).
removed=pk[10]
pk2=[pk[k] if k!=10 else 0 for k in range(max_k)]
S2=sum(pk2)
for k in range(max_k):
    pk2[k]/=S2
mean2=sum(k*pk2[k] for k in range(max_k))
second2=sum(k*k*pk2[k] for k in range(max_k))
q2=(second2-mean2)/mean2
print('original mean',mean,'q',q)
print('after removing k10 mean',mean2,'q2',q2,'removed fraction',removed)
