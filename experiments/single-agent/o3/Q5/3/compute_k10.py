
import scipy.special as sp, numpy as np
r=3
p=0.5
max_k=200
pk=[sp.comb(k+r-1,k)*(p**r)*((1-p)**k) for k in range(max_k)]
S=sum(pk)
pk=[x/S for x in pk]
mean=sum(k*pk[k] for k in range(max_k))
second=sum(k*k*pk[k] for k in range(max_k))
q=(second-mean)/mean
# remove all degree 10 nodes
pk2=[pk[k] if k!=10 else 0 for k in range(max_k)]
S2=sum(pk2)
pk2=[x/S2 for x in pk2]
mean2=sum(k*pk2[k] for k in range(max_k))
second2=sum(k*k*pk2[k] for k in range(max_k))
q2=(second2-mean2)/mean2
print(mean,second,q)
print(mean2,second2,q2)
print('fraction removed', pk[10])