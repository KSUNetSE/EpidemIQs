
import numpy as np, networkx as nx, scipy.sparse as sparse, os, math, json, random
from collections import Counter
np.random.seed(42)
N=10000
# Negative binomial parameters to get mean 3, variance 6 -> r=1.5, p=1/3
r=1.5
p=1/3
# sample degrees
degrees=np.random.negative_binomial(r, p, size=N)
# ensure no zero-degree improbable? keep zeros allowed but configuration model may isolate nodes.
# But sum of degrees must be even
if degrees.sum()%2==1:
    degrees[0]+=1
print('mean',degrees.mean())
print('second moment',np.mean(degrees**2))
mean_k=degrees.mean()
second_moment=np.mean(degrees**2)
q=(second_moment-mean_k)/mean_k
print('q',q)
