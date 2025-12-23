
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
P10=pk[10]

def q_eff(x):
    m=(mean - x*P10*10) / (1 - x*P10)
    s=(second - x*P10*100) / (1 - x*P10)
    return (s - m)/m
xs=np.linspace(0,1,10001)
thr=None
for x in xs:
    if q_eff(x)<1:
        thr=x
        break
print('thr',thr)