
import scipy.special as sp
r=3
p=0.5
max_k=200
pk=[sp.comb(k+r-1,k)*(p**r)*((1-p)**k) for k in range(max_k)]
# normalize
S=sum(pk)
pk=[x/S for x in pk]
mean=sum(k*pk[k] for k in range(max_k))
second=sum(k*k*pk[k] for k in range(max_k))
q=(second-mean)/mean
print('original mean',mean,'second',second,'q',q)
# probability k=10
prob10=pk[10]
# after removal
pk2=pk.copy()
removed=pk2[10]
pk2[10]=0
S2=sum(pk2)
pk2=[x/S2 for x in pk2]
mean2=sum(k*pk2[k] for k in range(max_k))
second2=sum(k*k*pk2[k] for k in range(max_k))
q2=(second2-mean2)/mean2
print('after removal mean',mean2,'second',second2,'q2',q2)
print('prob10',prob10)
