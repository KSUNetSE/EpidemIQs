
import os, numpy as np, scipy.sparse as sparse, scipy.linalg as la
import networkx as nx
from pathlib import Path
output_dir=os.path.join(os.getcwd(),'output')
A_path=os.path.join(output_dir,'network_layer_A.npz')
B_path=os.path.join(output_dir,'network_layer_B.npz')
A=sparse.load_npz(A_path).toarray()
B=sparse.load_npz(B_path).toarray()
# leading eigenvectors
wA, vA=la.eig(A)
idxA=np.argmax(wA.real)
vecA=vA[:,idxA].real
vecA=vecA/np.linalg.norm(vecA)

wB, vB=la.eig(B)
idxB=np.argmax(wB.real)
vecB=vB[:,idxB].real
vecB=vecB/np.linalg.norm(vecB)

cos_overlap=abs(np.dot(vecA,vecB))
