
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

from math import comb
r=3
p=0.5
# compute distribution for k up to 40
P=[comb(k+r-1,k)*p**r*(1-p)**k for k in range(41)]
P_sum=sum(P)
P=[pk/P_sum for pk in P]
z=sum(k*P[k] for k in range(len(P)))
second=sum(k*k*P[k] for k in range(len(P)))
print('mean', z, 'second', second)
P10=P[10]
print('P10',P10)

# function to compute R0' after vaccinating fraction x of degree 10 nodes

def Rprime(x):
    f=x*P10
    z_prime=(z - x*P10*10)/(1-f)
    second_prime=(second - x*P10*100)/(1-f)
    q_prime=(second_prime - z_prime)/z_prime
    return q_prime  # transmissibility same as originally (T=1)

# find x such that q'<1
import numpy as np
xs=np.linspace(0,1,101)
vals=[Rprime(x) for x in xs]
for x,v in zip(xs,vals):
    if v<1:
        print('x',x,'Rprime',v)
        break
