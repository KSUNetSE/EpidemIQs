
# Reasoning for Scenario 2:
# - Aggregated (static, weighted) network. SIR model with same parameters, 1 random seed, 100 sims, over 100 steps.
# - Using FastGEMF as per static scenario, loading network from .npz file.

import os
import numpy as np
import scipy.sparse as sparse
import fastgemf as fg
import pandas as pd

N = 1000
T = 100
nsim = 100
per_contact_infect_prob = 0.777
recovery_prob = 0.2
curdir = os.getcwd()
# Load static weighted network
agg_net_path = '/Users/hosseinsamaei/phd/epidemiqs/output/aggregated-network.npz'
G_csr = sparse.load_npz(agg_net_path)
assert G_csr.shape == (N, N), f"Adjacency matrix shape mismatch: {G_csr.shape}"

# --- Define SIR model schema for FastGEMF ---
SIR_model_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network_layer')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I', network_layer='contact_network_layer', rate='beta')
)

SIR_instance = (
    fg.ModelConfiguration(SIR_model_schema)
      .add_parameter(beta=per_contact_infect_prob, gamma=recovery_prob)
      .get_networks(contact_network_layer=G_csr)
)

# --- Initial condition: single seed infectious, all others S ---
# FastGEMF expects a vector of states (0=S, 1=I, 2=R)
seed_idx = np.random.choice(N)
X0 = np.zeros(N, dtype=int)
X0[seed_idx] = 1
initial_condition = {'exact': X0}

# --- Simulation ---
variation_type = "90ci"
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition,
                   stop_condition={'time': T}, nsim=nsim)
sim.run()

time, state_count, statecounts_lower_upper_bands = sim.get_results(variation_type=variation_type)

simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f"{SIR_model_schema.compartments[i]}"] = state_count[i, :]
    simulation_results[f"{SIR_model_schema.compartments[i]}_{variation_type}_lower"] = statecounts_lower_upper_bands[0,i]
    simulation_results[f"{SIR_model_schema.compartments[i]}_{variation_type}_upper"] = statecounts_lower_upper_bands[1,i]
data = pd.DataFrame(simulation_results)
out_csv_path = os.path.join(curdir, "output", "results-12.csv")
data.to_csv(out_csv_path, index=False)

sim.plot_results(time, state_count, variation_type=variation_type, show_figure=False, save_figure=True,
                 title="SIR Aggregated Static Weighted Network", save_path=os.path.join(curdir, "output", "results-12.png"))

# Epidemic metrics (final size, peak prevalence, t_peak)
I_mean = state_count[1, :]  # infected mean
R_mean = state_count[2, :]  # recovered mean
Ntot = float(N)
final_size_mean = R_mean[-1]/Ntot
peak_prev_mean = np.max(I_mean)/Ntot
peak_t = np.argmax(I_mean)

summary = {'final_size_mean': float(final_size_mean), 'peak_prev_mean': float(peak_prev_mean), 't_peak_mean': float(peak_t)}
# shape of state_count (nodes, time), paths
