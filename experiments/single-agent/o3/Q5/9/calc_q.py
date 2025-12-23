
import math, numpy as np
z=3
p=[math.exp(-z)*z**k/math.factorial(k) for k in range(0,100)]
mean=sum(k*p[k] for k in range(len(p)))
second=sum(k*k*p[k] for k in range(len(p)))
q=(second-mean)/mean
print(mean, second, q)