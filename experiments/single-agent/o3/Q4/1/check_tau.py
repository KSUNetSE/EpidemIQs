
import os, numpy as np, scipy.sparse as sparse, pandas as pd, networkx as nx
from numpy.linalg import eigvals
from numpy.random import default_rng
output_dir=os.path.join(os.getcwd(),'output')
G_A=sparse.load_npz(os.path.join(output_dir,'layerA.npz'))
G_B1=sparse.load_npz(os.path.join(output_dir,'layerB1.npz'))
G_B2=sparse.load_npz(os.path.join(output_dir,'layerB2.npz'))

lambda1_A=eigvals(G_A.toarray()).real.max()
lambda1_B2=eigvals(G_B2.toarray()).real.max()

beta_factor=3.0
beta1=beta_factor/lambda1_A
beta2=beta_factor/lambda1_B2

delta1=1.0
delta2=1.0

print(beta1*lambda1_A, beta2*lambda1_B2)