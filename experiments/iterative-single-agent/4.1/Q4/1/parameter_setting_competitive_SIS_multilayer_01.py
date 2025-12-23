
import numpy as np
params = {}

# Simulation parameters
# Use analytical results: want tau1, tau2 > threshold, but not so large as to cause dominance
# Largest eigenvalues for adjacency as threshold, use power iteration for largest eigenvalue
import scipy.sparse as sp
A = sp.load_npz('output/layer_A-01.npz')
B = sp.load_npz('output/layer_B-01.npz')
def largest_eigenvalue(M):
    v = np.random.rand(M.shape[0])
    v = v / np.linalg.norm(v)
    for _ in range(10):
        v = M @ v
        v = v / np.linalg.norm(v)
    return float(v @ (M @ v))
lambdaA = largest_eigenvalue(A)
lambdaB = largest_eigenvalue(B)

# Set infection and recovery rates such that tau > 1/lambda for both
# Use tau1 = 1.5/threshold1, tau2 = 1.5/threshold2 for both slightly above threshold

delta1 = 1.0
beta1 = 1.5 * (1/lambdaA) * delta1

delta2 = 1.0
beta2 = 1.5 * (1/lambdaB) * delta2

# Store parameters
params['beta1'] = beta1
params['delta1'] = delta1
params['beta2'] = beta2
params['delta2'] = delta2
params['lambdaA'] = lambdaA
params['lambdaB'] = lambdaB
params['tau1'] = beta1/delta1
params['tau2'] = beta2/delta2
params