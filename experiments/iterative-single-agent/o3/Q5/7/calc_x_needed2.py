
from math import comb
r=3
p=0.5
P=[comb(k+r-1,k)*p**r*(1-p)**k for k in range(1001)]
S=sum(P)
P=[pk/S for pk in P]
z=sum(k*P[k] for k in range(len(P)))
second=sum(k*k*P[k] for k in range(len(P)))
P10=P[10]

def q_after(x):
    f=x*P10
    if f>=1:
        return 0
    z_prime=(z - x*10*P10)/(1-f)
    second_prime=(second - x*100*P10)/(1-f)
    q_prime=(second_prime - z_prime)/z_prime
    return q_prime

low=0; high=1
for _ in range(60):
    mid=(low+high)/2
    if q_after(mid) > 1:
        low=mid
    else:
        high=mid
print('x', high, 'fraction of degree-10')
print('total vaccinated f', high*P10)
