
import scipy.special as sp
r=3
p=0.5
max_k=200
pk=[sp.comb(k+r-1,k)*(p**r)*((1-p)**k) for k in range(max_k)]
S=sum(pk)
for k in range(len(pk)):
    pk[k]/=S
mean=sum(k*pk[k] for k in range(max_k))
second=sum(k*k*pk[k] for k in range(max_k))
q=(second-mean)/mean
pk10=pk[10]
mean2=(mean - 10*pk10)/(1-pk10)
second2=(second - 100*pk10)/(1-pk10)
q2=(second2-mean2)/mean2
print(mean, second, q, pk10)
print(mean2, second2, q2)
