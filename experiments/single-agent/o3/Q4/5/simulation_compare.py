
import fastgemf as fg, os, numpy as np, scipy.sparse as sparse, pandas as pd
from scipy.sparse.linalg import eigs

output_dir = os.path.join(os.getcwd(), 'output')
A_csr = sparse.load_npz(os.path.join(output_dir, 'network_A.npz'))
B_same = sparse.load_npz(os.path.join(output_dir, 'network_B_same.npz'))
B_orig = sparse.load_npz(os.path.join(output_dir, 'network_B.npz'))
B_low = sparse.load_npz(os.path.join(output_dir, 'network_B_lowcorr.npz'))

# Function to compute beta from tau and lambda1

def leading_lambda(csr):
    return eigs(csr.astype(float), k=1, which='LR', return_eigenvectors=False)[0].real

lamA = leading_lambda(A_csr)
lam_same = leading_lambda(B_same)
lam_orig = leading_lambda(B_orig)
lam_low = leading_lambda(B_low)

delta1 = delta2 = 1.0
tau = 1.3  # above threshold
beta1 = tau/lamA * delta1  # constant for all scenarios

betas2 = {
    'same': tau/lam_same,
    'orig': tau/lam_orig,
    'low': tau/lam_low
}

# initial condition 5% each infected randomly but disjoint same across scenarios
N = A_csr.shape[0]
rand = np.random.RandomState(555)
perm = rand.permutation(N)
I1_nodes = perm[:int(0.05*N)]
I2_nodes = perm[int(0.05*N):int(0.1*N)]
state_base = np.zeros(N, dtype=int)
state_base[I1_nodes] = 1
state_base[I2_nodes] = 2

results_paths = {}
for idx, (key, B_csr) in enumerate([('same',B_same),('orig',B_orig),('low',B_low)], start=21):
    schema = (fg.ModelSchema('CompetitiveSIS')
                .define_compartment(['S','I1','I2'])
                .add_network_layer('A')
                .add_network_layer('B')
                .add_edge_interaction('inf1', from_state='S', to_state='I1', inducer='I1', network_layer='A', rate='beta1')
                .add_edge_interaction('inf2', from_state='S', to_state='I2', inducer='I2', network_layer='B', rate='beta2')
                .add_node_transition('rec1', from_state='I1', to_state='S', rate='delta1')
                .add_node_transition('rec2', from_state='I2', to_state='S', rate='delta2'))
    config = (fg.ModelConfiguration(schema)
                .add_parameter(beta1=beta1, beta2=betas2[key], delta1=delta1, delta2=delta2)
                .get_networks(A=A_csr, B=B_csr))
    sim = fg.Simulation(config, initial_condition={'exact':state_base}, stop_condition={'time':600}, nsim=5)
    sim.run()
    fig_path = os.path.join(output_dir, f'results-{idx}.png')
    sim.plot_results(show_figure=False, save_figure=True, save_path=fig_path)
    time, state_count, *_ = sim.get_results()
    df = pd.DataFrame({'time':time, 'S':state_count[0], 'I1':state_count[1], 'I2':state_count[2]})
    csv_path = os.path.join(output_dir, f'results-{idx}.csv')
    df.to_csv(csv_path, index=False)
    results_paths[key] = {'fig':fig_path, 'csv':csv_path, 'final_I1':int(state_count[1,-1]), 'final_I2':int(state_count[2,-1])}

results_paths