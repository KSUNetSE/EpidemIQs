
# Simulation 1: ER multiplex, maximally correlated layers (i=1, j=1)
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd
np.random.seed(22)  # For reproducibility

# 1. Load networks for maximally correlated ER multiplex
curdir = os.getcwd()
G_A = sparse.load_npz(os.path.join(curdir, 'output', 'layerA-ERMux.npz'))
G_B = sparse.load_npz(os.path.join(curdir, 'output', 'layerB-ERMux-maxcorr.npz')) # copy, rho=1

N = G_A.shape[0]

# 2. Define the competitive bi-virus SIS model on multiplex
model_schema = (
    fg.ModelSchema("CompBiVirusSIS")
    .define_compartment(['S', 'I1', 'I2'])
    .add_network_layer('layerA')
    .add_network_layer('layerB')
    .add_edge_interaction(
        name='virus1_infect', from_state='S', to_state='I1', inducer='I1',
        network_layer='layerA', rate='beta1')
    .add_node_transition(
        name='virus1_recover', from_state='I1', to_state='S', rate='delta1')
    .add_edge_interaction(
        name='virus2_infect', from_state='S', to_state='I2', inducer='I2',
        network_layer='layerB', rate='beta2')
    .add_node_transition(
        name='virus2_recover', from_state='I2', to_state='S', rate='delta2')
)

# 3. Model parameterization
beta1, delta1, beta2, delta2 = 0.03431, 0.25, 0.03275, 0.25
model_instance = (
    fg.ModelConfiguration(model_schema)
    .add_parameter(beta1=beta1, delta1=delta1, beta2=beta2, delta2=delta2)
    .get_networks(layerA=G_A, layerB=G_B)
)

# 4. Initial condition: 2% nodes to I1, 2% to I2, rest S (no overlap)
X0 = np.zeros(N, dtype=int)  # 0: S, 1: I1, 2: I2
idx = np.arange(N)
np.random.shuffle(idx)
nI1 = int(0.02 * N);
nI2 = int(0.02 * N)
I1_nodes, I2_nodes = idx[:nI1], idx[nI1:nI1+nI2]
X0[I1_nodes] = 1
X0[I2_nodes] = 2  # rest is S
initial_condition = {'exact': X0}

# 5. Simulate
sim = fg.Simulation(model_instance, initial_condition=initial_condition, stop_condition={'time': 800}, nsim=40) # 40 replications
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))
# Save results (time series)
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
for i in range(state_count.shape[0]):
    simulation_results[f'{model_schema.compartments[i]}'] = state_count[i, :]
df = pd.DataFrame(simulation_results)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)