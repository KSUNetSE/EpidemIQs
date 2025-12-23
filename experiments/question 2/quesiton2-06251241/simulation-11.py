
import os
import fastgemf as fg
import scipy.sparse as sparse
import pandas as pd
import numpy as np

# SIR Model schema: per-contact edge transmission (beta), node recovery (gamma)
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact')
    .add_node_transition(
        name="recovery",
        from_state='I',
        to_state='R',
        rate='gamma'
    )
    .add_edge_interaction(
        name="infection",
        from_state='S',
        to_state='I',
        inducer='I',
        network_layer='contact',
        rate='beta'
    )
)

# Scenario-specific parameters
network_path = os.path.join(os.getcwd(), 'output', 'er-network.npz')
beta, gamma = 0.02995, 0.1

G_csr = sparse.load_npz(network_path)

# Model configuration
cfg = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact=G_csr)
)

# Initial condition: 99% S, 1% I (10/1000), 0% R, random seeding.
initial_condition = {'percentage': {'S':99, 'I':1, 'R':0}}

sr = 75  # Number of stochastic realizations
sim = fg.Simulation(cfg, initial_condition=initial_condition, stop_condition={'time': 365}, nsim=sr)
sim.run()

# Save plot and results
plot_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
data_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
sim.plot_results(show_figure=False, save_figure=True, save_path=plot_path)

time, state_count, *_ = sim.get_results()
results = {'time': time}
for i, comp in enumerate(SIR_schema.compartments):
    results[f'{comp}'] = state_count[i, :]
pd.DataFrame(results).to_csv(data_path, index=False)

# For chain-of-thought trace:
print('Simulation completed for ER network, R0>1. Data and plot saved.')
print('Number of nodes:', G_csr.shape[0], 'Edges:', int(G_csr.nnz/2))
