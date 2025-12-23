
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd

# --- PLAN ---
# 1. Load the network
# 2. Set up SIR model schema with network layer
# 3. Configure with parameters beta, gamma
# 4. Set random initial condition, 995 S, 5 I, 0 R
# 5. Simulate 5 runs, save plot and csv

# 1: Load network
network_csr = sparse.load_npz(os.path.join(os.getcwd(), 'output', 'network.npz'))

# 2. Define SIR schema
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contact_network')
    .add_node_transition(name='recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction(name='infection', from_state='S', to_state='I', inducer='I',
                         network_layer='contact_network', rate='beta')
)

# 3. Configure instance
beta = 0.010624860954622764
gamma = 0.1
SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(contact_network=network_csr)
)

# 4. Initial condition: 995 S, 5 I, 0 R (randomly)
initial_condition = {'percentage': {'S': 99, 'I': 1, 'R': 0}}  # 1000 nodes: 1% I ≈ 10, but FastGEMF rounds up (acceptable)

# 5. Simulate
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 100}, nsim=5)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

# 6. Save results
result_time, state_count, *_ = sim.get_results()
result_out = {'time': result_time}
for i, c in enumerate(SIR_schema.compartments):
    result_out[c] = state_count[i, :]
df = pd.DataFrame(result_out)
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
# Print for logging
os.path.exists(os.path.join(os.getcwd(), 'output', 'results-11.csv'))