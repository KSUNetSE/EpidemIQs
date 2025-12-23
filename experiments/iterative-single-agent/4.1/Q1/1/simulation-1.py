
import numpy as np
import os
import fastgemf as fg
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

# Set the import parameters manually
beta = 0.25  # base transmission rate per infectious neighbor
sigma = 1/3  # exposure to infection (latent period ~3 days)
gamma = 1/4  # recovery period (~4 days)
R0 = 2.5

mean_deg_hom = 10.282
k2_hom = 116.524
mean_deg_het = 5.982
k2_het = 83.668

# For S-I-E-R: beta_eff = R0 * gamma / ( (<k^2> - <k>)/<k> )
def get_network_beta(R0, gamma, k, k2):
    q = (k2 - k)/k
    return R0 * gamma / q

beta_hom = get_network_beta(R0, gamma, mean_deg_hom, k2_hom)
beta_het = get_network_beta(R0, gamma, mean_deg_het, k2_het)

# Save table for report
output_table = pd.DataFrame(
    {'Network': ['Homogeneous (ER)', 'Heterogeneous (BA)'],
     'mean_degree': [mean_deg_hom, mean_deg_het],
     'k2': [k2_hom, k2_het],
     'beta': [beta_hom, beta_het]}
)
output_table.to_csv(os.path.join(os.getcwd(), 'output', 'network_parameters.csv'), index=False)

print('beta_hom', beta_hom)
print('beta_het', beta_het)

# Load networks
G_hom = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_hom.npz'))
G_het = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network_het.npz'))

# Model Schema SEIR
SEIR_schema = (
    fg.ModelSchema("SEIR")
    .define_compartment(["S", "E", "I", "R"])
    .add_network_layer("contact_network_layer")
    .add_edge_interaction(
        name='infection',
        from_state='S', to_state='E', inducer='I',
        network_layer='contact_network_layer', rate='beta'
    )
    .add_node_transition(
        name='progression', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(
        name='recovery', from_state='I', to_state='R', rate='gamma')
)

# Configurations hom & het
SEIR_hom = (
    fg.ModelConfiguration(SEIR_schema)
    .add_parameter(beta=float(beta_hom), sigma=sigma, gamma=gamma)
    .get_networks(contact_network_layer=G_hom)
)
SEIR_het = (
    fg.ModelConfiguration(SEIR_schema)
    .add_parameter(beta=float(beta_het), sigma=sigma, gamma=gamma)
    .get_networks(contact_network_layer=G_het)
)
# Initial condition: 5 infected (random), 995 susceptible
ic_dict = {'percentage': {'S': 99, 'E': 0, 'I': 1, 'R': 0}}

# Simulations (10 runs each)
sim_hom = fg.Simulation(SEIR_hom, initial_condition=ic_dict, stop_condition={'time': 120}, nsim=10)
sim_het = fg.Simulation(SEIR_het, initial_condition=ic_dict, stop_condition={'time': 120}, nsim=10)
sim_hom.run()
sim_hom.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-1-1.png'))
# Save CSV
H_time, H_state_count, *_ = sim_hom.get_results()
data = {'time': H_time}
for i, comp in enumerate(SEIR_schema.compartments):
    data[comp] = H_state_count[i, :]
df = pd.DataFrame(data)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-1-1.csv'), index=False)

sim_het.run()
sim_het.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-1-2.png'))
Het_time, Het_state_count, *_ = sim_het.get_results()
data = {'time': Het_time}
for i, comp in enumerate(SEIR_schema.compartments):
    data[comp] = Het_state_count[i, :]
df = pd.DataFrame(data)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-1-2.csv'), index=False)

# For reference - output
(beta_hom, beta_het)