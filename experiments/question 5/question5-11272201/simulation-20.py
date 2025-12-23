
# Reasoning (Step 3/5: Random vaccination near, but below threshold: 74%)
# - i=2, j=0 for random vaccination arm at v=74%, as 'just-below-phase-transition' check.
# - Assign 74% of nodes as R (immune), 1% of all as I (random among S), remainder S.
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
IC = np.zeros(N, dtype=int)
rng = np.random.default_rng(44)
# assign 74% random nodes as R
idxR = rng.choice(N, size=int(0.75*N), replace=False)
IC[idxR] = 2
# from non-R pool, assign 1% of N as I
avail = np.where(IC==0)[0]
idxI = rng.choice(avail, size=int(0.01*N), replace=False)
IC[idxI] = 1
model_inst = (fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=3.2, gamma=1.0)
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
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-21.csv'), index=False)
sim.plot_results(time, state_count, variation_type='90ci', show_figure=False, save_figure=True, title="Random vaccination 74% (just below threshold)", save_path=os.path.join(os.getcwd(), 'output', 'results-20.png'))
{'IC': IC, 'idxR': idxR, 'idxI': idxI}