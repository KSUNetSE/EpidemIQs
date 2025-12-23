
import math
r=3
p=0.5
k=10
def nb_pmf(k,r,p):
    from math import comb
    return comb(k+r-1,k)*(p**r)*((1-p)**k)
print(nb_pmf(10,3,0.5))