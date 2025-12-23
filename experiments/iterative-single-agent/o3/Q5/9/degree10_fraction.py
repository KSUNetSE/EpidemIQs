
import scipy.sparse as sparse, os, networkx as nx, numpy as np, collections, math
net=sparse.load_npz(os.path.join(os.getcwd(),'output','network.npz'))
# convert to networkx graph? easier to compute degree distribution from CSR counts
k = np.array(net.sum(axis=1)).flatten().astype(int)
vals=collections.Counter(k)
print('N=',len(k))
print('degree 10 count',vals[10])
print('fraction',vals[10]/len(k))
