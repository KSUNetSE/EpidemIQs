
# compute gamma for B4
import os, scipy.sparse as sparse, numpy as np, scipy
from scipy.sparse import linalg as spla
A=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_A.npz'))
B4=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_B4.npz'))
va=spla.eigs(A,k=1,which='LR')[1][:,0]
vb=spla.eigs(B4,k=1,which='LR')[1][:,0]
va=va/np.linalg.norm(va)
vb=vb/np.linalg.norm(vb)
cos=(abs(va.conj().T@vb))**2
cos