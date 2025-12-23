
# --- CHAIN OF THOUGHT Step 3/4 ---
# For completeness, generate and test synthetic core-periphery network for kc=1.0 (if the necessary .npz/.csv files are missing),
# since this would give maximal core-core density and should yield interesting behavior. 
# This will check autonomy to construct the test network, if needed.
# We use SBM for this purpose (10 core, 90 periphery, specified densities).
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import os

N = 100
n_core = int(0.1*N)
n_peri = N - n_core
kc = 1.0  # maximal core-core
kp = 0.075
kcp = 0.3

# Stochastic block model prob matrix
probs = np.full((2,2), kcp)  # default
probs[0,0] = kc  # core-core
temp = np.array(
    ([n_core, n_peri], [n_core, n_peri])
)
size = [n_core, n_peri]
probs[1,1] = kp  # peri-peri
rs = np.random.RandomState(99)

# Generate SBM graph
G = nx.stochastic_block_model(size, probs, seed=rs, sparse=True)
A_csr = nx.to_scipy_sparse_array(G, format='csr')
# Assign groups: core (0...n_core-1), periphery (n_core...N-1)
groups = np.array(['core']*n_core + ['periphery']*n_peri)
group_df = pd.DataFrame({'node': np.arange(N), 'group': groups})

net_path = os.path.join(os.getcwd(), 'output', 'core-periphery-network-kc1.00.npz')
groups_path = os.path.join(os.getcwd(), 'output', 'core-periphery-groups-kc1.00.csv')
sp.save_npz(net_path, A_csr)
group_df.to_csv(groups_path, index=False)
_return = {'kc_1_net_path': net_path, 'kc_1_group_path': groups_path}
