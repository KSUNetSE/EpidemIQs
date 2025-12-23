
import numpy as np, scipy.sparse as sparse, networkx as nx, os
from scipy.sparse import linalg as spla
A=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_A.npz'))
B=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_B.npz'))
va=spla.eigs(A,k=1,which='LR')[1][:,0]
vb=spla.eigs(B,k=1,which='LR')[1][:,0]
cosine=abs((va.conj().T@vb)/(np.linalg.norm(va)*np.linalg.norm(vb)))
cosine