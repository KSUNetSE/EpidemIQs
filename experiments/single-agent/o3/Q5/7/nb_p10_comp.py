
from math import comb
r=3
p=0.5
k=10
P10=comb(k+r-1,k)* (p**r)*((1-p)**k)
print(P10)
