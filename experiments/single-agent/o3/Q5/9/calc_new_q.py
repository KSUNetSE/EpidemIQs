
import math, numpy as np
z=3
max_k=30
P=[math.exp(-z)*z**k/math.factorial(k) for k in range(max_k+1)]
# Remove degree 10
P_removed=P.copy()
P_removed[10]=0
norm=1-P[10]
P_new=[p/norm for p in P_removed]
# compute mean degree and mean excess
z_new=sum(k*P_new[i] for i,k in enumerate(range(max_k+1)))
q_new=sum(k*(k-1)*P_new[i] for i,k in enumerate(range(max_k+1)))/z_new
z,q_new