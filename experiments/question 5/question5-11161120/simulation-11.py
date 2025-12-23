
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import random

# 1. Load the network
network_path = os.path.join('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]

# 2. Model schema: SIR
beta = 0.893
gamma = 1.0
sir_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('L')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='L', rate='beta')
)

# 3. Model configuration
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(L=G_csr)
)

# 4. Initial condition (random vaccination)
vacc_frac = 0.75
random.seed(30)
np.random.seed(30)
vaccinated = set(np.random.choice(N, int(N * vacc_frac), replace=False))

# Remaining are S or eligible for infection
remaining = list(set(range(N)) - vaccinated)
assert len(remaining) == int(N * 0.25)

# Infect 5 random unvaccinated
infected = set(np.random.choice(remaining, 5, replace=False))
susceptible = list(set(remaining) - infected)

# Assign node states: S=0, I=1, R=2
# All vaccinated are considered immune (put in state R directly so they play no role)
X0 = np.zeros(N, dtype=int)
X0[list(vaccinated)] = 2  # Immune (R)
X0[list(infected)] = 1    # Infectious (I)
X0[list(susceptible)] = 0 # Susceptible (S)
# Post check
assert (np.sum(X0==1) == 5)
assert (np.sum(X0==2) == int(N * vacc_frac))
assert (np.sum(X0==0) == (N - 5 - int(N * vacc_frac)))

init_cond = {'exact': X0}

# 5. Run simulation
nsim = 100
sim = fg.Simulation(sir_instance, initial_condition=init_cond, stop_condition={'time': 180}, nsim=nsim)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-11.png'))

# 6. Save results
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
# Compartments: S=0, I=1, R=2
for i, comp in enumerate(['S','I','R']):
    simulation_results[comp] = state_count[i,:]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'), index=False)
