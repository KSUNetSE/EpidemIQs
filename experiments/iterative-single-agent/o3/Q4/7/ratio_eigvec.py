
import os, numpy as np, scipy.sparse as sp, scipy.linalg as la
output_dir=os.path.join(os.getcwd(),'output')
A=sp.load_npz(os.path.join(output_dir,'network_layer_A.npz')).toarray()
B=sp.load_npz(os.path.join(output_dir,'network_layer_B.npz')).toarray()
# leading eigenvectors
valsA, vecsA = la.eig(A)
idxA=np.argmax(valsA.real)
vecA=vecsA[:,idxA].real
valsB, vecsB = la.eig(B)
idxB=np.argmax(valsB.real)
vecB=vecsB[:,idxB].real
vecA=vecA/np.linalg.norm(vecA)
vecB=vecB/np.linalg.norm(vecB)
phi=max(vecA/vecB)
psi=max(vecB/vecA)
