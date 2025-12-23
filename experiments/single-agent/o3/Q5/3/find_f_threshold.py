
import scipy.special as sp, numpy as np
r=3
p=0.5
max_k=200
pk=[sp.comb(k+r-1,k)*(p**r)*((1-p)**k) for k in range(max_k)]
S=sum(pk)
for k in range(max_k):
    pk[k]/=S
mean=sum(k*pk[k] for k in range(max_k))
second=sum(k*k*pk[k] for k in range(max_k))
print('q original',(second-mean)/mean)
prob10=pk[10]
print('prob10',prob10)

def effective_q(f):
    # remove fraction f of degree10
    removed_nodes=f*prob10
    mean_new=(mean - 10*removed_nodes)/(1-removed_nodes)
    second_new=(second - 100*removed_nodes)/(1-removed_nodes)
    return (second_new - mean_new)/mean_new

fs=np.linspace(0,1,101)
for f in fs:
    if effective_q(f)<1:
        print('threshold f',f)
        break
else:
    print('never crosses')