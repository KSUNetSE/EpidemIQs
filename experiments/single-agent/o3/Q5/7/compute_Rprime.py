
from math import comb
r=3; p=0.5
z=3
# compute distribution up to 100
P=[comb(k+r-1,k)*p**r*(1-p)**k for k in range(101)]
# normalize (should sum to 1)
import numpy as np
print('sum', sum(P))
mean=sum(k*P[k] for k in range(101))
second=sum(k*k*P[k] for k in range(101))
print(mean, second)
P10=P[10]
f=P10  # fraction vaccinated
z_prime=(z-10*P10)/(1-P10)
second_prime=(second-100*P10)/(1-P10)
q_prime=(second_prime - z_prime)/z_prime
R_prime=q_prime*1  # since transmissibility unchanged, R0' = q'*T, and T tuned so original q=4, R0=4; T =1 presumably? Actually R0=q=4? Wait R0=4 and q=4 so T=1? But not, R0 = T*q. They say R0=4 and q=4, implies T=1. So threshold is just q. But continue.
print('P10',P10, 'z_prime', z_prime, 'second_prime', second_prime, 'q_prime', q_prime)
