
# Reasoning:
# First, let's test a single scenario (i=0, j=0): lowest transmission for both viruses, initial condition I1 seed only. This checks if schema, multiplex, exclusive infection, and output work as intended before running the full batch. 

import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

# --- Parameters for (i=0, j=0) ---
params_grid = [
    [0.05747, 1.0, 0.0652, 1.0],  # i=0: beta1, delta1, beta2, delta2
    # others omitted for this test
]
i, j = 0, 0 # First scenario/IC index
beta1, delta1, beta2, delta2 = params_grid[i]

# --- Network loading ---
layerA_path = os.path.join(os.getcwd(), 'output', 'network-layerA.npz')
layerB_path = os.path.join(os.getcwd(), 'output', 'network-layerB.npz')
G_A = sparse.load_npz(layerA_path)
G_B = sparse.load_npz(layerB_path)

# --- Model schema with 2 layers and exclusive infection ---
SIS2_model = (
    fg.ModelSchema('CompetitiveSIS')
    .define_compartment(['S', 'I1', 'I2'])
    # S -(I1 via A)-> I1
    .add_network_layer('layerA')
    .add_edge_interaction(
        name='infection1', from_state='S', to_state='I1', inducer='I1', network_layer='layerA', rate='beta1'
    )
    # S -(I2 via B)-> I2
    .add_network_layer('layerB')
    .add_edge_interaction(
        name='infection2', from_state='S', to_state='I2', inducer='I2', network_layer='layerB', rate='beta2'
    )
    # I1 -> S
    .add_node_transition(
        name='rec1', from_state='I1', to_state='S', rate='delta1')
    # I2 -> S
    .add_node_transition(
        name='rec2', from_state='I2', to_state='S', rate='delta2')
)

# --- Model instance (tie networks to correct edge-interactions) ---
model_instance = (
    fg.ModelConfiguration(SIS2_model)
    .add_parameter(beta1=beta1, delta1=delta1, beta2=beta2, delta2=delta2)
    .get_networks(layerA=G_A, layerB=G_B)
)
N = G_A.shape[0]

# --- Initial condition: 1% I1, rest S (IC case j=0) ---
X0 = np.zeros(N, dtype=int) # 0 = S, 1 = I1, 2 = I2
# Randomly pick 10 nodes as I1
rng = np.random.default_rng(202406) # reproducibility
I1_inds = rng.choice(N, size=10, replace=False)
X0[I1_inds] = 1
initial_condition = { 'exact': X0 }

# --- Simulation setup ---
sr = 50 # stochastic realizations (adjust if needed for insight)
sim = fg.Simulation(model_instance, initial_condition=initial_condition, stop_condition={'time': 200}, nsim=sr)

sim.run()

# Plot/save output
plot_path = os.path.join(os.getcwd(), 'output', f'results-{i}{j}.png')
sim.plot_results(show_figure=False, save_figure=True, save_path=plot_path)

# Save full time series as CSV
# FastGEMF returns (time, state_count, ...)
time, state_count, *_ = sim.get_results()
sim_results = { 'time': time }
for idx, comp in enumerate(SIS2_model.compartments):
    sim_results[comp] = state_count[idx, :]

csv_path = os.path.join(os.getcwd(), 'output', f'results-{i}{j}.csv')
pd.DataFrame(sim_results).to_csv(csv_path, index=False)

# Network structure for output
network_details = f"LayerA: N={N}, edges={G_A.nnz//2}; LayerB: N={N}, edges={G_B.nnz//2}"

# Return key metadata for summary/aggregation step
output_info = {
    'simulation_details': [
        f'Competitive SIS (exclusive, multiplex, anti-hub-overlap): param set {i} (beta1={beta1}, delta1={delta1}, beta2={beta2}, delta2={delta2}), IC-{j} (1% I1, rest S). Ran {sr} stochastic realizations on negative-hub-overlap multiplex (A/B, N=1000, e.g. edges: {G_A.nnz//2}/{G_B.nnz//2}). Time horizon=200.'
    ],
    'stored_result_path': {csv_path:'Raw compartment time series (ensemble-avg) for this run'},
    'Plot_path': {plot_path:network_details},
    'success_of_simulation': True,
}
