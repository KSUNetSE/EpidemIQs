
import numpy as np, networkx as nx, scipy.sparse as sparse, os
out=os.path.join(os.getcwd(),'output')
G = sparse.load_npz(os.path.join(out,'network.npz'))
Gnx = nx.from_scipy_sparse_array(G)
ks = np.array([d for _,d in Gnx.degree()])
N=len(ks)
A = ks[ks!=10].sum()/N  # since P_k = count/N; sum_{k≠10} P_k k = total deg excluding k10 / N
B = np.sum(ks[ks!=10]*(ks[ks!=10]-1))/N
P10 = np.mean(ks==10)
threshold = 1 - (A - B)/(80*P10)
print(P10,A,B,threshold)
