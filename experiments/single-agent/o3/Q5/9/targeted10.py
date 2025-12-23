
import math
mean=3
P=[math.exp(-mean)*mean**k/math.factorial(k) for k in range(0,50)]
P10=P[10]
S1=sum(k*P[k] for k in range(len(P)))
S2=sum(k*k*P[k] for k in range(len(P)))
print('P10',P10)

for x in [i/10 for i in range(11)]:
    denom=1-x*P10
    S1p=(S1 - x*10*P10)/denom
    S2p=(S2 - x*100*P10)/denom
    qp=(S2p - S1p)/S1p
    R0p=4*qp/((S2 - S1)/S1)  # R0 scaling
    print(x, qp, R0p)
