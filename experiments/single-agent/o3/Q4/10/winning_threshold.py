
import os, numpy as np, scipy.sparse as sparse
from scipy.sparse import linalg as spla

A=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_A.npz'))
B=sparse.load_npz(os.path.join(os.getcwd(),'output','layer_B.npz'))

lambdaA=float(np.real(spla.eigs(A,k=1,which='LR',return_eigenvectors=False)[0]))
lambdaB=float(np.real(spla.eigs(B,k=1,which='LR',return_eigenvectors=False)[0]))
va=spla.eigs(A,k=1,which='LR')[1][:,0]
va=va/np.linalg.norm(va)
vb=spla.eigs(B,k=1,which='LR')[1][:,0]
vb=vb/np.linalg.norm(vb)

gamma=(abs(va.conj().T@vb))**2  # squared cosine

tau1=beta1*lambdaA/delta1 # effective adoption ratio
beta1=0.10401059578111396
delta1=1.0
beta2=0.13639308317445792
delta2=1.0
tau1=beta1*lambdaA/delta1
tau2=beta2*lambdaB/delta2

# Survival threshold is 1, since tau>1 means adoption effective > threshold

# Winning threshold for virus1 (eqn from paper): tau1 = (1 + (tau2 -1)*gamma)
# in normalized by lambda? Actually formula: T1* = (1/lambda1(A)) [ 1 + (tau2 lambdaB -1)*gamma ]? Wait: convert to tau. Let's assume with tau defined using same normalization by lambda: tau1_w = 1 + (tau2 -1)*gamma

T1_w=1 + (tau2 -1)*gamma
T2_w=1 + (tau1 -1)*gamma

coexist_possible = (tau1>1 and tau2>1) and (tau1<T1_w) and (tau2<T2_w)

data={'lambdaA':lambdaA,'lambdaB':lambdaB,'gamma':gamma,'tau1':tau1,'tau2':tau2,'T1_w':T1_w,'T2_w':T2_w,'coexist_pred':coexist_possible}
