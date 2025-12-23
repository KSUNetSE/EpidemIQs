
import pandas as pd, os
output_dir=os.path.join(os.getcwd(),'output')
rand_df=pd.read_csv(os.path.join(output_dir,'results-11.csv'))
rand_final=rand_df['cum_infected'].iloc[-1]
rand_total_nodes=5000
rand_attack=rand_final/rand_total_nodes

tar_df=pd.read_csv(os.path.join(output_dir,'results-12.csv'))
# total nodes after removal
import scipy.sparse as sp, networkx as nx
csr=sp.load_npz(os.path.join(output_dir,'network.npz'))
import numpy as np
G_full=nx.from_scipy_sparse_array(csr)
N_full=G_full.number_of_nodes()
import collections
k10=[n for n,d in G_full.degree() if d==10]
N_tar=N_full - len(k10)

tar_final=tar_df['cum_infected'].iloc[-1]
attack_tar=tar_final/N_tar
result_metrics={'rand_final':rand_final,'rand_attack':rand_attack,'tar_final':tar_final,'tar_attack':attack_tar,'N_rand':rand_total_nodes,'N_tar':N_tar}
