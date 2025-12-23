
import math, numpy as np
mu=3
r=3
P=[math.comb(k+r-1,k)* (r/(r+mu))**r * (mu/(r+mu))**k for k in range(0,100)]
P=[x/sum(P) for x in P]
S1=sum(k*P[k] for k in range(len(P)))
S2=sum(k*k*P[k] for k in range(len(P)))
print('mean',S1)
print('q_orig',(S2-S1)/S1)
P10=P[10]
print('P10',P10)
S1_orig=S1
S2_orig=S2
q_orig=(S2-S1)/S1

crit_x=None
for x in np.linspace(0,1,10001):
    denom=1 - x*P10
    S1p=(S1 - x*10*P10)/denom
    S2p=(S2 - x*100*P10)/denom
    qp=(S2p - S1p)/S1p
    if qp < 1:
        crit_x=x
        break
print('critical_x',crit_x)
if crit_x is None:
    print('Not possible to reach R<1 by only vaccinating degree10')
    crit_fraction = None
else:
    crit_fraction=crit_x*P10
    print('fraction of total population', crit_fraction)
