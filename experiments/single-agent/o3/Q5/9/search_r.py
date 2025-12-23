
import math, numpy as np
mu=3
best=None
for r in np.linspace(0.2,10,100):
    P=[math.comb(int(k+r-1),k)*(r/(r+mu))**r*(mu/(r+mu))**k for k in range(0,100)]
    P=[x/sum(P) for x in P]
    S1=sum(k*P[k] for k in range(len(P)))
    S2=sum(k*k*P[k] for k in range(len(P)))
    q=(S2-S1)/S1
    if abs(q-4)<0.05:
        P10=P[10]
        best=(r,q,P10)
print(best)
