
import math
mu=3
r=3
p0=(r/(r+mu))**r
P=[math.comb(k+r-1,k)* (r/(r+mu))**r * (mu/(r+mu))**k for k in range(0,100)]
mean=sum(k*P[k] for k in range(len(P)))
second=sum(k*k*P[k] for k in range(len(P)))
q=(second-mean)/mean
print(mean, q)
P10=P[10]
print('P10',P10)
