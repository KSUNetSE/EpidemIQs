
from math import comb
P10=66/8192
z=3
second=15
z_prime=(z-10*P10)/(1-P10)
second_prime=(second-100*P10)/(1-P10)
q_prime=(second_prime - z_prime)/z_prime
print(P10, z_prime, second_prime, q_prime)
