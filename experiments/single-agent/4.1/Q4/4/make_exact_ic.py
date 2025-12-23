
# Redefine initial condition as percent for FastGEMF: we need sum to 1 or 100 (integers), for N=1000
# 997 S, 3 I -> S: 99.7%, I: 0.3%, rounded to S: 100, I: 0 for 100 (or S: 99, I: 1 if N=100)
# Instead, directly use 'exact' for full state vector
import scipy.sparse as sparse
import numpy as np
import os
import fastgemf as fg
import pandas as pd

network_path = os.path.join(os.getcwd(), 'output', 'network.npz')
N = 1000
# Create the exact initial state vector: 997 S (0), 3 I (1)
X0 = np.zeros(N, dtype=int)
X0[:3] = 1
np.random.shuffle(X0)
initial_condition = {'exact': X0}
np.save(os.path.join(os.getcwd(), 'output', 'X0.npy'), X0)
# Confirm state vector composition
unique, counts = np.unique(X0, return_counts=True)
init_counts = dict(zip(unique.tolist(), counts.tolist()))
init_counts