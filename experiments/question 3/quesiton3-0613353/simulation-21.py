
# simulation-21.py: Static weighted network SIR, beta=0.00042, gamma=0.1
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import os
import pandas as pd

# 1. Load static weighted contact network (CSR, weights = mean freq per step)
static_net_path = os.path.join(os.getcwd(), 'output', 'exp3-static-weighted.npz')
G_csr = sparse.load_npz(static_net_path)
N = G_csr.shape[0]

# 2. Model definition: SIR with edge-dependent infection (beta*w_ij), I->R recovery
SIR_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('contacts')
    .add_edge_interaction(
        name='weighted_infection',
        from_state='S',
        to_state='I',
        inducer='I',
        network_layer='contacts',
        rate='beta'
    )
    .add_node_transition(
        name='recovery',
        from_state='I',
        to_state='R',
        rate='gamma'
    )
)

# 3. Model configuration with *per-edge* rate (FastGEMF interprets rate parameter as multiplying edge weights if edge matrix populated)
SIR_instance = (
    fg.ModelConfiguration(SIR_schema)
    .add_parameter(beta=0.00042, gamma=0.1)
    .get_networks(contacts=G_csr)
)

# 4. Initial condition: 10 random infected, rest susceptible
np.random.seed(4321)
X0 = np.zeros(N, dtype=int)
infected_init = np.random.choice(N, 10, replace=False)
X0[infected_init] = 1
initial_condition = {'exact': X0}

# 5. Simulation, with robust number of stochastic realizations
sr = 120
sim = fg.Simulation(SIR_instance, initial_condition=initial_condition, stop_condition={'time': 2000}, nsim=sr)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-21.png'))

time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i in range(state_count.shape[0]):
    simulation_results[f'{SIR_schema.compartments[i]}'] = state_count[i,:]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-21.csv'), index=False)