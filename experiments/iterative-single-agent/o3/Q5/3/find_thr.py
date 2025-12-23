
import scipy.special as sp, numpy as np
r=3
p=0.5
max_k=200
pk=[sp.comb(k+r-1,k)*(p**r)*((1-p)**k) for k in range(max_k)]
S=sum(pk)
for k in range(max_k):
    pk[k]/=S
mean=sum(k*pk[k] for k in range(max_k))
second=sum(k*k*pk[k] for k in range(max_k))
p_d=pk[10]
print('p_d',p_d)

mean0=mean
second0=second

def q_eff(x):
    if x>=1: x=1-1e-8
    m=(mean0 - x*p_d*10) / (1 - x*p_d)
    s=(second0 - x*p_d*100)/ (1 - x*p_d)
    q=(s - m)/m
    return q

xs=np.linspace(0,1,10001)
thr=None
for x in xs:
    if q_eff(x)<1:
        thr=x
        break
print('threshold',thr)
