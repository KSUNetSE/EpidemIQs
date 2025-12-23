
import math
from collections import defaultdict

r=3
p=0.5

def nb_prob(k,r,p):
    # P(k)=C(k+r-1,k)*p^r*(1-p)^k
    coeff=math.comb(k+r-1,k)
    return coeff*(p**r)*((1-p)**k)

P=[nb_prob(k,r,p) for k in range(0,30)]
print(sum(P))
mean=sum(k*P[k] for k in range(len(P)))
var=sum((k-mean)**2*P[k] for k in range(len(P)))
print('mean',mean,'var',var)
P10=nb_prob(10,r,p)
print('P10',P10)
