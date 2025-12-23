
import numpy as np, networkx as nx, scipy.sparse as sparse, os, math, pandas as pd, random
# load degs array from earlier environment? We'll re-sample as before
N=1000000
r=3; p=0.5
rng=np.random.default_rng(1)
max_deg=100
probs=[math.comb(k+r-1,k)*(1-p)**k * p**r for k in range(max_deg)]
probs=np.array(probs); probs/=probs.sum()
ks=rng.choice(np.arange(len(probs)),size=N,p=probs)
mean_k=ks.mean()
q=( (ks**2).mean() - mean_k)/mean_k
print(mean_k,q)
# fraction with k==10
p10=(ks==10).mean()
print('p10',p10)
