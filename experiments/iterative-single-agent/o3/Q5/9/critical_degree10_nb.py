
import math, numpy as np
mu=3
r=3
# create neg bin distribution
P=[math.comb(k+r-1,k)*(r/(r+mu))**r*(mu/(r+mu))**k for k in range(0,100)]
S1=sum(k*P[k] for k in range(len(P)))
S2=sum(k*k*P[k] for k in range(len(P)))
print('mean',S1,'q', (S2-S1)/S1)
P10=P[10]
print('P10',P10)
for x in np.linspace(0,1,101):
    denom=1 - x*P10
    S1p=(S1 - x*10*P10)/denom
    S2p=(S2 - x*100*P10)/denom
    qp=(S2p - S1p)/S1p
    if qp < 1:
        print('critical x',x,'fraction of population vaccinated',x*P10)
        break
