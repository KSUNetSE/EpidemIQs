
# Write the simulation code for random vaccination (75%) to file.
import os
code = '''
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
'''
with open(os.path.join(os.getcwd(), 'output', 'simulation-11.py'), 'w') as f:
    f.write(code)

# Now execute the written simulation for (random vaccination, 75%) and collect outputs.
import os
os.system(f'python {os.path.join(os.getcwd(), "output", "simulation-11.py")}')

# After running, summarize result/plot locations and stats.
import scipy.sparse as sparse
G_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
simulated_model_details = "Random vaccination at threshold (75%): SIR over configuration-model, N=10,000, β=0.893, γ=1, 75% vaccinated pre-epidemic, 5 initially infected, rest susceptible."
stored_result_path = {os.path.join(os.getcwd(), 'output', 'results-11.csv'): 'Trajectory results for random vaccination (75%)'}
plot_path = {os.path.join(os.getcwd(), 'output', 'results-11.png'): 'SIR epidemic time course (random vaccination, 75% coverage)'}
network_stats = {'nodes': G_csr.shape[0], 'edges': G_csr.nnz//2}
simulated_model_details, stored_result_path, plot_path, network_stats
# Simulation 2-1: Targeted vaccination — all degree-10 nodes vaccinated
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd

# 1. Load the network and degrees
network_path = os.path.join('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]
# Compute degrees
degrees = np.array(G_csr.sum(axis=1)).flatten()

# 2. Identify degree-10 nodes
deg10_nodes = np.where(degrees == 10)[0]
p10 = len(deg10_nodes) / N

# 3. Model schema: SIR
beta = 0.893
gamma = 1.0
sir_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('L')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='L', rate='beta')
)

# 4. Model configuration
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(L=G_csr)
)

# 5. Initial condition: all degree-10 nodes vaccinated, 5 random others infected
vaccinated = set(deg10_nodes)
vacc_frac = len(vaccinated) / N

# Remaining eligible for infection
remaining = list(set(range(N)) - set(vaccinated))
np.random.seed(42)
infected = set(np.random.choice(remaining, 5, replace=False))
susceptible = set(remaining) - infected

# Assign node states: S=0, I=1, R=2
X0 = np.zeros(N, dtype=int)
X0[list(vaccinated)] = 2  # Immune (R)
X0[list(infected)] = 1    # Infectious (I)
X0[list(susceptible)] = 0 # Susceptible (S)
assert (np.sum(X0 == 1) == 5)
assert (np.sum(X0 == 2) == len(vaccinated))

init_cond = {'exact': X0}

# 6. Run simulation
nsim = 100
sim = fg.Simulation(sir_instance, initial_condition=init_cond, stop_condition={'time': 180}, nsim=nsim)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-21.png'))

# 7. Save results
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i, comp in enumerate(['S','I','R']):
    simulation_results[comp] = state_count[i,:]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-21.csv'), index=False)

# Summarize results after targeted degree-10 vaccination simulation
import scipy.sparse as sparse
G_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
simulated_model_details = "Targeted vaccination (degree-10): all degree-10 nodes vaccinated (~9.9%), SIR over configuration-model, N=10,000, β=0.893, γ=1, 5 initially infected."
stored_result_path = {os.path.join(os.getcwd(), 'output', 'results-21.csv'): 'Trajectory results for targeted vaccination (degree-10)'}
plot_path = {os.path.join(os.getcwd(), 'output', 'results-21.png'): 'SIR epidemic time course (targeted vaccination, all degree-10 nodes)'}
network_stats = {'nodes': G_csr.shape[0], 'edges': G_csr.nnz//2}
simulated_model_details, stored_result_path, plot_path, network_stats
# (Optional Control) Simulate baseline (no vaccination)
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd

# 1. Load the network
G_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
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
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(L=G_csr)
)
# 3. Initial conditions: 5 infected, rest susceptible
np.random.seed(99)
X0 = np.zeros(N, dtype=int)
I_idx = np.random.choice(N, 5, replace=False)
X0[I_idx] = 1
init_cond = {'exact': X0}

# 4. Run simulation
sim = fg.Simulation(sir_instance, initial_condition=init_cond, stop_condition={'time': 180}, nsim=100)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-00.png'))

# 5. Save results
time, state_count, *_ = sim.get_results()
simulation_results = {'time': time, 'S': state_count[0,:], 'I': state_count[1,:], 'R': state_count[2,:]}
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-00.csv'), index=False)

# Summarize baseline simulation (no vaccination)
import scipy.sparse as sparse
G_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
simulated_model_details = "Baseline control (no vaccination): SIR model on configuration-model network, N=10,000, 5 initial infections."
stored_result_path = {os.path.join(os.getcwd(), 'output', 'results-00.csv'): 'SIR trajectory for baseline no-vaccination scenario'}
plot_path = {os.path.join(os.getcwd(), 'output', 'results-00.png'): 'SIR epidemic time course for baseline/control'}
network_stats = {'nodes': G_csr.shape[0], 'edges': G_csr.nnz//2}
simulated_model_details, stored_result_path, plot_path, network_stats
# Scenario (1-2): Random vaccination, below threshold (65%)
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import random

network_path = os.path.join('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]

beta = 0.893
gamma = 1.0
sir_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('L')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='L', rate='beta')
)
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(L=G_csr)
)
vacc_frac = 0.65
random.seed(31)
np.random.seed(31)
vaccinated = set(np.random.choice(N, int(N * vacc_frac), replace=False))
remaining = list(set(range(N)) - vaccinated)
assert len(remaining) == int(N * 0.35)
infected = set(np.random.choice(remaining, 5, replace=False))
susceptible = list(set(remaining) - infected)
X0 = np.zeros(N, dtype=int)
X0[list(vaccinated)] = 2
X0[list(infected)] = 1
X0[list(susceptible)] = 0
assert (np.sum(X0==1) == 5)
assert (np.sum(X0==2) == int(N * vacc_frac))
assert (np.sum(X0==0) == (N - 5 - int(N * vacc_frac)))
init_cond = {'exact': X0}
nsim = 100
sim = fg.Simulation(sir_instance, initial_condition=init_cond, stop_condition={'time': 180}, nsim=nsim)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-12.png'))
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i, comp in enumerate(['S','I','R']):
    simulation_results[comp] = state_count[i,:]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-12.csv'), index=False)

# Summarize scenario (1-2): random vaccination, 65%
import scipy.sparse as sparse
G_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
simulated_model_details = "Random vaccination below threshold (65%): SIR configuration-model, N=10,000, β=0.893, γ=1, 65% vaccinated, 5 infected."
stored_result_path = {os.path.join(os.getcwd(), 'output', 'results-12.csv'): 'Trajectory results for random vaccination (65%)'}
plot_path = {os.path.join(os.getcwd(), 'output', 'results-12.png'): 'SIR epidemic time course (random vaccination, 65%)'}
network_stats = {'nodes': G_csr.shape[0], 'edges': G_csr.nnz//2}
simulated_model_details, stored_result_path, plot_path, network_stats
# Scenario (1-3): Random vaccination, above threshold (85%)
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd
import random

network_path = os.path.join('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]

beta = 0.893
gamma = 1.0
sir_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('L')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='L', rate='beta')
)
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(L=G_csr)
)
vacc_frac = 0.85
random.seed(32)
np.random.seed(32)
vaccinated = set(np.random.choice(N, int(N * vacc_frac), replace=False))
remaining = list(set(range(N)) - vaccinated)
assert len(remaining) == int(N * 0.15)
infected = set(np.random.choice(remaining, 5, replace=False))
susceptible = list(set(remaining) - infected)
X0 = np.zeros(N, dtype=int)
X0[list(vaccinated)] = 2
X0[list(infected)] = 1
X0[list(susceptible)] = 0
assert (np.sum(X0==1) == 5)
assert (np.sum(X0==2) == int(N * vacc_frac))
assert (np.sum(X0==0) == (N - 5 - int(N * vacc_frac)))
init_cond = {'exact': X0}
nsim = 100
sim = fg.Simulation(sir_instance, initial_condition=init_cond, stop_condition={'time': 180}, nsim=nsim)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-13.png'))
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i, comp in enumerate(['S','I','R']):
    simulation_results[comp] = state_count[i,:]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-13.csv'), index=False)

# Summarize scenario (1-3): random vaccination, 85%
import scipy.sparse as sparse
G_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
simulated_model_details = "Random vaccination above threshold (85%): SIR configuration-model, N=10,000, β=0.893, γ=1, 85% vaccinated, 5 infected."
stored_result_path = {os.path.join(os.getcwd(), 'output', 'results-13.csv'): 'Trajectory results for random vaccination (85%)'}
plot_path = {os.path.join(os.getcwd(), 'output', 'results-13.png'): 'SIR epidemic time course (random vaccination, 85%)'}
network_stats = {'nodes': G_csr.shape[0], 'edges': G_csr.nnz//2}
simulated_model_details, stored_result_path, plot_path, network_stats
# Scenario (2-2): Targeted degree-10, below threshold (vaccinate 7% => 70% of degree-10 nodes)
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd

network_path = os.path.join('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]
degrees = np.array(G_csr.sum(axis=1)).flatten()
deg10_nodes = np.where(degrees == 10)[0]
# Target: vaccinate 7% of all nodes, i.e. (7/9.9 ≈ 70%) of deg10 cohort
n_vacc = int(N * 0.07)
np.random.seed(43)
vaccinated_deg10 = np.random.choice(deg10_nodes, n_vacc, replace=False)
vaccinated = set(vaccinated_deg10)
vacc_frac = len(vaccinated) / N
remaining = list(set(range(N)) - vaccinated)
infected = set(np.random.choice(remaining, 5, replace=False))
susceptible = set(remaining) - infected
X0 = np.zeros(N, dtype=int)
X0[list(vaccinated)] = 2
X0[list(infected)] = 1
X0[list(susceptible)] = 0
init_cond = {'exact': X0}

# Model config
beta = 0.893
gamma = 1.0
sir_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('L')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='L', rate='beta')
)
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(L=G_csr)
)
nsim = 100
sim = fg.Simulation(sir_instance, initial_condition=init_cond, stop_condition={'time': 180}, nsim=nsim)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-22.png'))
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i, comp in enumerate(['S','I','R']):
    simulation_results[comp] = state_count[i,:]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-22.csv'), index=False)

# Summarize scenario (2-2): targeted deg-10 vaccination, 7% of nodes (~70% deg-10 cohort)
import scipy.sparse as sparse
G_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
simulated_model_details = "Targeted, below threshold: 7% of nodes (≈70% of deg-10 cohort) vaccinated; SIR, N=10,000, β=0.893, γ=1, 5 initial infections."
stored_result_path = {os.path.join(os.getcwd(), 'output', 'results-22.csv'): 'Trajectory (7% targeted, deg-10)'}
plot_path = {os.path.join(os.getcwd(), 'output', 'results-22.png'): 'SIR epidemic, targeted (7% deg-10 nodes)'}
network_stats = {'nodes': G_csr.shape[0], 'edges': G_csr.nnz//2}
simulated_model_details, stored_result_path, plot_path, network_stats
# Scenario (2-3): Targeted deg-10 vaccination, above threshold (vaccinate 12% — covers 100% deg-10 + spillover)
import os
import fastgemf as fg
import scipy.sparse as sparse
import numpy as np
import pandas as pd

network_path = os.path.join('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
G_csr = sparse.load_npz(network_path)
N = G_csr.shape[0]
degrees = np.array(G_csr.sum(axis=1)).flatten()
deg10_nodes = np.where(degrees == 10)[0]
n_deg10 = len(deg10_nodes)
n_vacc = int(N * 0.12)
if n_vacc <= n_deg10:
    vaccinated = set(np.random.choice(deg10_nodes, n_vacc, replace=False))
else:
    vaccinated = set(deg10_nodes)
    # Fill spillover with random others
    n_extra = n_vacc - n_deg10
    rest_nodes = list(set(range(N)) - set(deg10_nodes))
    vaccinated.update(np.random.choice(rest_nodes, n_extra, replace=False))
vacc_frac = len(vaccinated) / N
remaining = list(set(range(N)) - vaccinated)
np.random.seed(44)
infected = set(np.random.choice(remaining, 5, replace=False))
susceptible = set(remaining) - infected
X0 = np.zeros(N, dtype=int)
X0[list(vaccinated)] = 2
X0[list(infected)] = 1
X0[list(susceptible)] = 0
init_cond = {'exact': X0}

# Model config
beta = 0.893
gamma = 1.0
sir_schema = (
    fg.ModelSchema("SIR")
    .define_compartment(['S', 'I', 'R'])
    .add_network_layer('L')
    .add_node_transition('recovery', from_state='I', to_state='R', rate='gamma')
    .add_edge_interaction('infection', from_state='S', to_state='I', inducer='I', network_layer='L', rate='beta')
)
sir_instance = (
    fg.ModelConfiguration(sir_schema)
    .add_parameter(beta=beta, gamma=gamma)
    .get_networks(L=G_csr)
)
nsim = 100
sim = fg.Simulation(sir_instance, initial_condition=init_cond, stop_condition={'time': 180}, nsim=nsim)
sim.run()
sim.plot_results(show_figure=False, save_figure=True, save_path=os.path.join(os.getcwd(), 'output', 'results-23.png'))
time, state_count, *_ = sim.get_results()
simulation_results = {}
simulation_results['time'] = time
for i, comp in enumerate(['S','I','R']):
    simulation_results[comp] = state_count[i,:]
data = pd.DataFrame(simulation_results)
data.to_csv(os.path.join(os.getcwd(), 'output', 'results-23.csv'), index=False)

# Summarize scenario (2-3): targeted deg-10 vaccination, 12% (all deg-10 + spillover)
import scipy.sparse as sparse
G_csr = sparse.load_npz('/Users/hosseinsamaei/phd/gemf_llm/output/network.npz')
simulated_model_details = "Targeted, above threshold: 12% total vaccination (all deg-10 + spillover); SIR, N=10,000, β=0.893, γ=1, 5 infected."
stored_result_path = {os.path.join(os.getcwd(), 'output', 'results-23.csv'): 'Trajectory (12% targeted, includes all deg-10)'}
plot_path = {os.path.join(os.getcwd(), 'output', 'results-23.png'): 'SIR epidemic, targeted (12%), all deg-10+some)'}
network_stats = {'nodes': G_csr.shape[0], 'edges': G_csr.nnz//2}
simulated_model_details, stored_result_path, plot_path, network_stats