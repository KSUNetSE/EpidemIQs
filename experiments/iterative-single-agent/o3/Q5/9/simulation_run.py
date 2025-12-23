
import os, fastgemf as fg, numpy as np, scipy.sparse as sparse, networkx as nx, random, pandas as pd, math

# Paths
net_path=os.path.join(os.getcwd(),'output','network.npz')
G_csr = sparse.load_npz(net_path)
N = G_csr.shape[0]

# compute q_emp from network
k = np.array(G_csr.sum(axis=1)).flatten()
mean_k = k.mean()
second = (k*k).mean()
q_emp = (second - mean_k)/mean_k
print('mean_k',mean_k,'q_emp',q_emp)

# Model schema
SIR_schema=(fg.ModelSchema('SIR').define_compartment(['S','I','R']).add_network_layer('contact').add_edge_interaction('infection',from_state='S',to_state='I',inducer='I',network_layer='contact',rate='beta').add_node_transition('recovery',from_state='I',to_state='R',rate='gamma'))

gamma = 1.0
beta = 4 * gamma / q_emp  # calibrate R0=4
print('beta',beta)
config=(fg.ModelConfiguration(SIR_schema).add_parameter(beta=beta,gamma=gamma).get_networks(contact=G_csr))
print(config)

# Helper to run simulation and save

def run_sim(label, vaccinated_indices):
    # label integer for file naming (11,12,...)
    X0 = np.zeros(N, dtype=int)  # S=0
    X0[vaccinated_indices] = 2  # R=2
    # choose infected among susceptible
    sus_indices = np.where(X0==0)[0]
    n_infected = max(1, int(0.01*len(sus_indices)))
    infected_indices = np.random.choice(sus_indices, size=n_infected, replace=False)
    X0[infected_indices] = 1  # I state id=1
    initial_condition = {'exact': X0}

    sim = fg.Simulation(config, initial_condition=initial_condition, stop_condition={'time':100}, nsim=20)
    sim.run()
    # save plot
    fig_path = os.path.join(os.getcwd(),'output',f'results-{label}.png')
    sim.plot_results(show_figure=False, save_figure=True, save_path=fig_path)
    # extract last run results
    time, state_counts, *_ = sim.get_results()
    data = {'time': time}
    for idx,comp in enumerate(SIR_schema.compartments):
        data[comp] = state_counts[idx,:]
    df = pd.DataFrame(data)
    csv_path = os.path.join(os.getcwd(),'output',f'results-{label}.csv')
    df.to_csv(csv_path, index=False)
    print('saved', csv_path, fig_path)

# Scenario 1: Random vaccination 75%
vacc_rand = np.random.choice(N, size=int(0.75*N), replace=False)
run_sim(11, vacc_rand)

# Scenario 2: Vaccinate all degree 10
G_deg = k.astype(int)
vacc_deg10 = np.where(G_deg==10)[0]
run_sim(12, vacc_deg10)
