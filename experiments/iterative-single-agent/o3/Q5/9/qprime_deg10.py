
import math, numpy as np
mean=3
# compute sums upto maybe 30
P=[math.exp(-mean)*mean**k/math.factorial(k) for k in range(0,50)]
S1=sum(k*P[k] for k in range(len(P)))
S2=sum(k*k*P[k] for k in range(len(P)))
P10=P[10]
print('P10',P10)
S1_base=S1
S2_base=S2

for x in np.linspace(0,1,11):
    v=x*P10
    denom = 1 - v
    S1p = (S1 - x*10*P10)/denom
    S2p = (S2 - x*100*P10)/denom
    qprime = (S2p - S1p)/S1p
    print(x, qprime)
