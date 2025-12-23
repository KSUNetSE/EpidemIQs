
# Reasoning step:
# Step 1: Load the SBM network from provided path and check size/structure.
# Step 2: Define SEIBR model in FastGEMF: S -> E -> I -> B -> R, with infection only from I.
# Step 3: Calibrate with given rates: beta = 0.3789, sigma = 2.0, gamma1 = 0.7, gamma2 = 0.5.
# Step 4: For initial condition: seed with 1 exposed node, all others S.
# Step 5: Run 100 stochastic simulations up to 50 days. Extract B(t), R(t) (for convalescent/C), and attack rate.
# Step 6: Save results to output/results-11.csv and output/results-11.png.

import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import os

N = 763
network_path = os.path.join(os.getcwd(), 'output', 'england-boarding-sbm.npz')
G_csr = sparse.load_npz(network_path)

# Define SEIBR model schema
SEIBR_schema = (
    fg.ModelSchema('SEIBR')
    .define_compartment(['S', 'E', 'I', 'B', 'R'])
    .add_network_layer('contact_network')
    .add_edge_interaction(
        name='infection', from_state='S', to_state='E', inducer='I',
        network_layer='contact_network', rate='beta'  # Transmission ONLY from I
    )
    .add_node_transition(name='latent_exit', from_state='E', to_state='I', rate='sigma')
    .add_node_transition(name='symptom_isolation', from_state='I', to_state='B', rate='gamma1')
    .add_node_transition(name='recover', from_state='B', to_state='R', rate='gamma2')
)

# Model parameters (per analysis)
params = {"beta": 0.3789, "sigma": 2.0, "gamma1": 0.7, "gamma2": 0.5}
SEIBR_model = (
    fg.ModelConfiguration(SEIBR_schema)
    .add_parameter(**params)
    .get_networks(contact_network=G_csr)
)

# Initial condition: one random node E=1, everyone else S
X0 = np.zeros(N, dtype=int)
seed = np.random.default_rng(seed=42).integers(0, N)  # reproducibility
X0[seed] = 1  # E=1 (.../compartments: S=0, E=1, I=2, B=3, R=4)
initial_condition = {"exact": X0}

# Simulation: 100 realizations, 50 days
sr = 100
sim = fg.Simulation(SEIBR_model, initial_condition=initial_condition, stop_condition={'time': 50}, nsim=sr)
sim.run()
variation_type = "90ci"
# Extract results
results = sim.get_results(variation_type=variation_type)
time, state_mean, [state_low, state_high] = results

# Build output table: time, S, E, I, B, R, and CI bands for B
out_dict = {'time': time}
comps = ['S', 'E', 'I', 'B', 'R']
for i, c in enumerate(comps):
    out_dict[c] = state_mean[i]
    out_dict[c+'_lower'] = state_low[i]
    out_dict[c+'_upper'] = state_high[i]
# For convalescent: track new R (delta R per day) as C
C = np.diff(state_mean[4], prepend=0)
out_dict['C'] = C

data = pd.DataFrame(out_dict)
result_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data.to_csv(result_path, index=False)

#Save plots (B(t) and R(t), with CI)
sim.plot_results(time, state_mean, variation_type=variation_type, show_figure=False, save_figure=True,
                 title="SEIBR SBM simulation: Beta=0.3789 (SBM clustering)", save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

# Compute AR, peak B (timing, height)
attack_rate = state_mean[4, -1] / N
peak_B_idx = np.argmax(state_mean[3])
peak_B_time = time[peak_B_idx]
peak_B_value = state_mean[3, peak_B_idx]

return_vars = [
    'result_path', 'attack_rate', 'peak_B_time', 'peak_B_value'
]

