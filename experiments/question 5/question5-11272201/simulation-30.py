
# Reasoning (Step 4/5: Targeted vaccination - all degree-10 nodes vaccinated)
# - i=3, j=0 for all degree-10 nodes as R (immune); 1% of remainder as I, rest S.
# - Use degree sequence from provided net, reproducible selection for 'I'.
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os
N = 10000
sr = 40
stop_time = 200
network_path = '/Users/hosseinsamaei/phd/epidemiqs/output/network-cm-mixeddegree.npz'
G_csr = sparse.load_npz(network_path)
SIR_schema = (fg.ModelSchema("SIR_network")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('layer')
    .add_node_transition(name='recover', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infect', from_state='S', to_state='I', inducer='I', network_layer='layer', rate='beta'))
# recover degree sequence
import networkx as nx
G = nx.from_scipy_sparse_array(G_csr)
degree_seq = np.array([d for n, d in G.degree()])
IC = np.zeros(N, dtype=int)  # S=0
idx10 = np.where(degree_seq==10)[0]
IC[idx10] = 2  # mark all degree-10 as R
# find available non-vaccinated nodes
avail = np.where(IC==0)[0]
rng = np.random.default_rng(45)
idxI = rng.choice(avail, size=int(0.01*N), replace=False)
IC[idxI] = 1
model_inst = (fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=1e9, gamma=1.0)
    .get_networks(layer=G_csr))
sim = fg.Simulation(model_inst, initial_condition={'exact': IC}, stop_condition={'time': stop_time}, nsim=sr)
sim.run()
time, state_count, statebands = sim.get_results(variation_type="90ci")
results = {'time': time}
cmpts = SIR_schema.compartments
for i in range(state_count.shape[0]):
    results[f"{cmpts[i]}"] = state_count[i, :]
    results[f"{cmpts[i]}_90ci_lower"] = statebands[0,i]
    results[f"{cmpts[i]}_90ci_upper"] = statebands[1,i]
data = pd.DataFrame(results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-30.csv'), index=False)
sim.plot_results(time, state_count, variation_type='90ci', show_figure=False, save_figure=True, title="Targeted vaccination (all degree-10)", save_path=os.path.join(os.getcwd(), 'output', 'results-30.png'))
{'IC': IC, 'idx10': idx10, 'idxI': idxI}