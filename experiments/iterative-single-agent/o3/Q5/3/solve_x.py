
import scipy.special as sp
r=3
p=0.5
max_k=200
pk=[sp.comb(k+r-1,k)*(p**r)*((1-p)**k) for k in range(max_k)]
S=sum(pk)
for k in range(max_k):
    pk[k]/=S
P10=pk[10]
mean=sum(k*pk[k] for k in range(max_k))
second=sum(k*k*pk[k] for k in range(max_k))
import numpy as np

def R_eff(x):
    # x fraction of k=10 nodes removed
    num=second - x*P10*100
    den=mean - x*P10*10
    # renormalize by remaining fraction
    frac=1 - x*P10
    num/=frac
    den/=frac
    q=(num - den)/den
    return q
# find x such that q<1
xs=np.linspace(0,1,101)
for x in xs:
    if R_eff(x)<1:
        print('threshold x',x)
        break
