
# Now BA, random IC, phi=0.01,0.02, 1 rep each, save results and plot (as pilot for concentrated topology scenario).
import os
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def si_threshold_sim(adjacency, phi, initial_I_idx, N=500, max_steps=100):
    if not isinstance(adjacency, csr_matrix):
        adjacency = csr_matrix(adjacency)
    state = np.zeros(N, dtype=int)
    state[initial_I_idx] = 1
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    S_list, I_list = [(state==0).sum()], [(state==1).sum()]
    steps = 0
    while True:
        susceptible = np.where(state==0)[0]
        new_infected = []
        for i in susceptible:
            k = degrees[i]
            if k == 0: continue
            neighbor_idxs = adjacency.getrow(i).indices
            m = sum(state[neighbor_idxs]==1)
            if k > 0 and m/k >= phi:
                new_infected.append(i)
        if not new_infected or steps >= max_steps:
            break
        state[new_infected] = 1
        S_list.append((state==0).sum())
        I_list.append((state==1).sum())
        steps += 1
    return np.array(S_list), np.array(I_list), steps

def get_random_ic(N, num_I, seed=None):
    rng = np.random.default_rng(seed)
    infected = rng.choice(N, size=num_I, replace=False)
    return infected

N = 500
phi_list = [0.01, 0.02]
num_I = 5
adj_path = '/Users/hosseinsamaei/phd/epidemiqs/output/ba-adjacency.npz'
adj = scipy.sparse.load_npz(adj_path)
adj = csr_matrix(adj)
results_dir = os.path.join(os.getcwd(),'output')
os.makedirs(results_dir, exist_ok=True)

scen = 3 # 3: BA random IC
stored_result_path = {}
Plot_path = {}
run_details = []
for j, phi in enumerate(phi_list,1):
    seed = 1888 + j
    initial_I_idx = get_random_ic(N, num_I, seed)
    Slist, Ilist, steps = si_threshold_sim(adj, phi, initial_I_idx, N=N, max_steps=100)
    df = pd.DataFrame({'step': np.arange(len(Slist)), 'S': Slist, 'I': Ilist})
    csv_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}0.csv')
    df.to_csv(csv_path, index=False)
    stored_result_path[csv_path] = f"BA, random IC, φ={phi:.2f}, rep 1. Final size: {Ilist[-1]}/500"
    plt.figure()
    plt.plot(df['step'], df['I']/N, label='Defaulted (I)')
    plt.plot(df['step'], df['S']/N, label='Susceptible (S)')
    plt.xlabel('Step')
    plt.ylabel('Fraction')
    plt.title(f'BA, random, φ={phi:.2f}, rep 1')
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}0.png')
    plt.savefig(fig_path)
    plt.close()
    Plot_path[fig_path] = f"BA random, φ={phi:.2f}, rep 1, time series"
    run_details.append(f"BA, random IC, φ={phi:.2f}, rep=1: final I = {Ilist[-1]}/500, steps = {len(df)-1}")
return_vars = ['run_details','stored_result_path','Plot_path']