
import math, itertools
r=3
p=0.5
from math import comb
P=[comb(k+r-1,k)*p**r*(1-p)**k for k in range(0,50)]
P10=P[10]
mean=sum(k*P[k] for k in range(50))
second=sum(k*k*P[k] for k in range(50))
print(mean, second)
z=mean
p10=P10
z_prime=(z-10*p10)/(1-p10)
second_prime=(second-100*p10)/(1-p10)
q_prime=(second_prime - z_prime)/z_prime
print(p10, z_prime, second_prime, q_prime)
