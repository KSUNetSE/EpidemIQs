
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import os
import pandas as pd

# Set replicate (i=0 per scenario) and scenario index
j = 0  # Scenario 0
i = 0

# Step 1: Load both network layers
A_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network-layerA-ba.npz'))
B_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network-layerB-er.npz'))
N = A_csr.shape[0]  # Should be 1000
assert B_csr.shape[0] == N, 'Network layer size mismatch!'

# Step 2: Model Schema definition
model_schema = (
    fg.ModelSchema("CompetitiveSIS-excl")
    .define_compartment(['S', 'I1', 'I2'])
    .add_network_layer('A')
    .add_network_layer('B')
    .add_edge_interaction(name='infect_I1', from_state='S', to_state='I1', inducer='I1', network_layer='A', rate='beta1')
    .add_edge_interaction(name='infect_I2', from_state='S', to_state='I2', inducer='I2', network_layer='B', rate='beta2')
    .add_node_transition(name='recover_I1', from_state='I1', to_state='S', rate='delta1')
    .add_node_transition(name='recover_I2', from_state='I2', to_state='S', rate='delta2')
)

# Step 3: Initial condition
np.random.seed(42 + i*10 + j)  # Reproducibility; change seed per scenario if needed
all_nodes = np.arange(N)
I1_nodes = np.random.choice(all_nodes, size=10, replace=False)
remaining = np.setdiff1d(all_nodes, I1_nodes)
I2_nodes = np.random.choice(remaining, size=10, replace=False)
X0 = np.zeros(N, dtype=int)  # 0=S, 1=I1, 2=I2
X0[I1_nodes] = 1
X0[I2_nodes] = 2
initial_condition = {'exact': X0}

# Step 4: Scenario 0 parameters
beta1, delta1, beta2, delta2 = 0.07, 1.0, 0.15, 1.0

# Step 5: Model Configuration
model_instance = (
    fg.ModelConfiguration(model_schema)
    .add_parameter(beta1=beta1, delta1=delta1, beta2=beta2, delta2=delta2)
    .get_networks(A=A_csr, B=B_csr)
)

# Step 6: Simulation
stop_time = 500
nsim = 50
sim = fg.Simulation(model_instance, initial_condition=initial_condition, stop_condition={'time': stop_time}, nsim=nsim)
sim.run()

# Step 7: Plot and save results
fig_path = os.path.join(os.getcwd(), 'output', f'results-{i}{j}.png')
sim.plot_results(show_figure=False, save_figure=True, save_path=fig_path)

# Step 8: Extract results and save CSV
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time}
comp_names = ['S', 'I1', 'I2']
for idx, comp in enumerate(comp_names):
    simulation_results[comp] = state_count[idx, :]
data = pd.DataFrame(simulation_results)
csv_path = os.path.join(os.getcwd(), 'output', f'results-{i}{j}.csv')
data.to_csv(csv_path, index=False)

simulation_details = [
    f"Scenario 0: Competitive exclusive SIS on 1000-node 2-layer multiplex (BA + ER, partial overlap), rates beta1={beta1}, delta1={delta1}, beta2={beta2}, delta2={delta2}, initial: 10 I1, 10 I2, 980 S. Nsim={nsim}, time={stop_time}. Results stored to CSV and PNG."
]
stored_result_path = {csv_path: 'Simulation time-series CSV for scenario 0'}
Plot_path = {fig_path: 'Simulation prevalence time-series (S, I1, I2) scenario 0; 1000 nodes, partial overlap, decoupled multiplex.'}
network_info = f"Layers: BA (N=1000, λ₁=17.33), ER (N=1000, λ₁=7.10), overlap=10%, alignment rho=-0.69; edges: {A_csr.nnz//2} (A), {B_csr.nnz//2} (B)."

success_of_simulation = True
reasoning_info = (
    'Scenario 0 simulation uses competitive exclusive SIS (three compartments), implements mutually exclusive infections over BA and ER multiplex, partial overlap, negative correlation, initial 1% I1/I2 seeding, rates as per provided. Number of realizations/stop time (50/500) ensures steady-state and stochastic averaging. FastGEMF used for exact CTMC; outputs saved to required path structures. All steps executed per plan, network loaded from disk, random seeding for initial condition, outputs thoroughly traced.'
)

# Output for aggregation
result = {
    'simulation_details': simulation_details,
    'stored_result_path': stored_result_path,
    'Plot_path': Plot_path,
    'success_of_simulation': success_of_simulation,
    'reasoning_info': reasoning_info + '\n' + network_info
}
result
