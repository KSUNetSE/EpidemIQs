
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import os

# Load the two network layers
path_A = os.path.join(os.getcwd(), 'output', 'network_A.npz')
path_B = os.path.join(os.getcwd(), 'output', 'network_B.npz')
A_csr = sparse.load_npz(path_A)
B_csr = sparse.load_npz(path_B)

# Parameters from previous execution
n = 500
kA = 7.38
kA2 = 61.656
kB = 7.62
kB2 = 65.688

# Spectral radii (approximate, use nx and scipy)
import networkx as nx
G_A = nx.from_scipy_sparse_array(A_csr)
G_B = nx.from_scipy_sparse_array(B_csr)
lambda1_A = np.max(np.abs(np.linalg.eigvals(nx.to_numpy_array(G_A))))
lambda1_B = np.max(np.abs(np.linalg.eigvals(nx.to_numpy_array(G_B))))

# Choose effective infection rates larger than threshold so that both survive (for test):
# tau1 = beta1/delta1 > 1/lambda1_A  ; tau2 = beta2/delta2 > 1/lambda1_B
beta1 = 0.13
beta2 = 0.13
delta1 = 0.09
delta2 = 0.09
# They give tau1 ~ 1.44 > 1/lambda1_A , tau2 ~ 1.44 > 1/lambda1_B

# Build ModelSchema: SI1SI2S (exclusive infection, 3 states: S, I1, I2)
model_schema = (
    fg.ModelSchema('CompetitiveSIS')
    .define_compartment(['S', 'I1', 'I2'])
    .add_network_layer('layer_A')
    .add_network_layer('layer_B')
    .add_node_transition(name='recov1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition(name='recov2', from_state='I2', to_state='S', rate='delta2')
    .add_edge_interaction(name='inf1', from_state='S', to_state='I1', inducer='I1', network_layer='layer_A', rate='beta1')
    .add_edge_interaction(name='inf2', from_state='S', to_state='I2', inducer='I2', network_layer='layer_B', rate='beta2')
)

# Configure model
model_instance = (
    fg.ModelConfiguration(model_schema)
    .add_parameter(beta1=beta1, delta1=delta1, beta2=beta2, delta2=delta2)
    .get_networks(layer_A=A_csr, layer_B=B_csr)
)

# Initial condition: random 2% for each virus, rest susceptible
init_I1 = int(n * 0.02)
init_I2 = int(n * 0.02)
init_S = n - init_I1 - init_I2
X0 = np.zeros(n, dtype=int) # map: S=0, I1=1, I2=2
X0[:init_I1] = 1
X0[init_I1:init_I1+init_I2] = 2
np.random.shuffle(X0)
initial_condition = {'exact': X0}

# Run simulation: 5 runs, up to 200 time units
sim = fg.Simulation(model_instance, initial_condition=initial_condition, stop_condition={'time': 200}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

# Save CSV results
import pandas as pd
time, state_count, *_ = sim.get_results()
sim_results = {'time': time}
for i, lab in enumerate(model_schema.compartments):
    sim_results[lab] = state_count[i, :]
pd.DataFrame(sim_results).to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)

results_out = {'sim_csv': os.path.join(os.getcwd(), 'output', 'results-11.csv'),
              'sim_img': os.path.join(os.getcwd(), 'output', 'results-11.png')}
results_out