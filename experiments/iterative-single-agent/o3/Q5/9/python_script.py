
import math
z=3
k=10
P10=math.exp(-z)*z**k/math.factorial(k)
P10
import math
def negbin_pmf(k,r,p):
    # r integer? can be real but compute with gamma.
    from math import comb
    return math.comb(k+r-1,k)*(1-p)**r * p**k
r=3
p=0.5
P10=negbin_pmf(10,r,0.5)
print(P10)
import math, scipy.stats, numpy as np
lam=3
k=10
P10=math.exp(-lam)*lam**k/math.factorial(k)
P10
import math
r=3
p=0.5
def negbin_pmf(k,r,p):
    # NB distribution of number of failures until r successes. Mean= pr/(1-p) ??? Wait depending.
    # We'll use paramization: P(k) = C(k+r-1,k)*(1-p)^r * p^k
    return math.comb(k + r -1, k) * (1-p)**r * p**k
P10=negbin_pmf(10,r,p)
P10
import math
r=3
p=0.5
def negbin_pmf(k,r,p):
    from math import comb
    return comb(k+r-1,k)*(1-p)**r * p**k
P10=negbin_pmf(10,r,p)
print('P10',P10)

import math
import numpy as np
z=3
# compute degree distribution up to 30
P=[math.exp(-z)*z**k/math.factorial(k) for k in range(0,60)]
# compute baseline
k1=sum(k*P[k] for k in range(len(P)))
ksq_minus_k=sum(k*(k-1)*P[k] for k in range(len(P)))
q=ksq_minus_k/k1
print('baseline q',q)
P10=P[10]
print('P10',P10)

def q_after(f):
    # fraction f of k=10 nodes removed.
    numerator=0.0
    denom=0.0
    for k,p in enumerate(P):
        if k==10:
            p_new=(1-f)*p
        else:
            p_new=p
        denom+=k*p_new
        numerator+=k*(k-1)*p_new
    # renormalize by (1-v) where v=f*P10 removed nodes
    v=f*P10
    denom=denom/(1-v)
    numerator=numerator/(1-v)
    return numerator/denom

for f in [0,0.25,0.5,0.75,1.0]:
    print(f,q_after(f))