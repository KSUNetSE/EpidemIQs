
import os, numpy as np, scipy.sparse as sparse, networkx as nx
A=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_A.npz'))
# get degrees of A to reorder
import networkx as nx
G_A=nx.from_scipy_sparse_array(A)
degs=np.array([d for _,d in G_A.degree()])
ordered=np.argsort(degs)  # low to high
reverse=np.argsort(-degs) # high to low
perm=np.zeros_like(ordered)
perm[reverse]=ordered  # map high degree to low degree
from scipy.sparse import csr_matrix
P=csr_matrix((np.ones(len(perm)),(perm, np.arange(len(perm)))),shape=(len(perm),len(perm)))
B4=P@A@P.T
sparse.save_npz(os.path.join(os.getcwd(),'output','layer_B4.npz'),B4)
# compute correlation eigenvectors
from scipy.sparse import linalg as spla
va=spla.eigs(A,k=1,which='LR')[1][:,0]
vb=spla.eigs(B4,k=1,which='LR')[1][:,0]
cos=abs((va.conj().T@vb)/(np.linalg.norm(va)*np.linalg.norm(vb)))
cos