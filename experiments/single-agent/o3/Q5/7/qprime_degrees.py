
from math import comb
r=3
p=0.5
# compute distribution upper 100
P=[comb(k+r-1,k)*p**r*(1-p)**k for k in range(101)]
P_sum=sum(P)
P=[pk/P_sum for pk in P]
z=sum(k*P[k] for k in range(len(P)))
second=sum(k*k*P[k] for k in range(len(P)))
P10=P[10]

def q_prime(x):
    f=x*P10
    if f>=1:
        return 0
    z_prime=(z - x*10*P10)/(1-f)
    second_prime=(second - x*100*P10)/(1-f)
    q=(second_prime - z_prime)/z_prime
    return q

for x in [0,0.25,0.5,0.75,1]:
    print(x, q_prime(x))
