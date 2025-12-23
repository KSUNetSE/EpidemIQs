
# Now scale up to cover the first five phi values and three reps for each scenario, random initial conditions (as a core demonstration). # This gives 2 networks × 5 phi × 3 reps = 30 simulations.
# (1) ER, random IC  (scen 1)
# (2) BA, random IC  (scen 3)
# Both models, first 5 phi (as in full grid, can scale up as last step for all phi). # Save all csv/plots as before.
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

phi_list = [0.01, 0.02, 0.03, 0.04, 0.05]
num_I = 5
num_reps = 3
N = 500
results_dir = os.path.join(os.getcwd(),'output')
os.makedirs(results_dir, exist_ok=True)

scen = 1
adj_path = '/Users/hosseinsamaei/phd/epidemiqs/output/er-adjacency.npz'
adj = scipy.sparse.load_npz(adj_path)
adj = csr_matrix(adj)
stored_result_path = {}
Plot_path = {}
run_details = []
for j, phi in enumerate(phi_list,1):
    for rep in range(num_reps):
        seed = 9999 + 10*j + rep
        initial_I_idx = get_random_ic(N, num_I, seed)
        Slist, Ilist, steps = si_threshold_sim(adj, phi, initial_I_idx, N=N, max_steps=100)
        df = pd.DataFrame({'step': np.arange(len(Slist)), 'S': Slist, 'I': Ilist})
        csv_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}{rep}.csv')
        df.to_csv(csv_path, index=False)
        stored_result_path[csv_path] = f"ER, random IC, φ={phi:.2f}, rep {rep+1}. Final size: {Ilist[-1]}/500"
        plt.figure()
        plt.plot(df['step'], df['I']/N, label='Defaulted (I)')
        plt.plot(df['step'], df['S']/N, label='Susceptible (S)')
        plt.xlabel('Step')
        plt.ylabel('Fraction')
        plt.title(f'ER, random, φ={phi:.2f}, rep {rep+1}')
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}{rep}.png')
        plt.savefig(fig_path)
        plt.close()
        Plot_path[fig_path] = f"ER random, φ={phi:.2f}, rep {rep+1}, time series"
        run_details.append(f"ER, random IC, φ={phi:.2f}, rep={rep+1}: final I = {Ilist[-1]}/500, steps = {len(df)-1}")
return_vars = ['run_details','stored_result_path','Plot_path']
# (2) BA, random IC
def si_threshold_sim(adjacency, phi, initial_I_idx, N=500, max_steps=100):
    from scipy.sparse import csr_matrix
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

import os
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
N = 500
phi_list = [0.01, 0.02, 0.03, 0.04, 0.05]
num_I = 5
num_reps = 3
results_dir = os.path.join(os.getcwd(),'output')
os.makedirs(results_dir, exist_ok=True)
scen = 3
adj_path = '/Users/hosseinsamaei/phd/epidemiqs/output/ba-adjacency.npz'
adj = scipy.sparse.load_npz(adj_path)
from scipy.sparse import csr_matrix
adj = csr_matrix(adj)
stored_result_path = {}
Plot_path = {}
run_details = []
for j, phi in enumerate(phi_list,1):
    for rep in range(num_reps):
        seed = 8888 + 10*j + rep
        initial_I_idx = get_random_ic(N, num_I, seed)
        Slist, Ilist, steps = si_threshold_sim(adj, phi, initial_I_idx, N=N, max_steps=100)
        df = pd.DataFrame({'step': np.arange(len(Slist)), 'S': Slist, 'I': Ilist})
        csv_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}{rep}.csv')
        df.to_csv(csv_path, index=False)
        stored_result_path[csv_path] = f"BA, random IC, φ={phi:.2f}, rep {rep+1}. Final size: {Ilist[-1]}/500"
        plt.figure()
        plt.plot(df['step'], df['I']/N, label='Defaulted (I)')
        plt.plot(df['step'], df['S']/N, label='Susceptible (S)')
        plt.xlabel('Step')
        plt.ylabel('Fraction')
        plt.title(f'BA, random, φ={phi:.2f}, rep {rep+1}')
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}{rep}.png')
        plt.savefig(fig_path)
        plt.close()
        Plot_path[fig_path] = f"BA random, φ={phi:.2f}, rep {rep+1}, time series"
        run_details.append(f"BA, random IC, φ={phi:.2f}, rep={rep+1}: final I = {Ilist[-1]}/500, steps = {len(df)-1}")
return_vars = ['run_details','stored_result_path','Plot_path']
# Now ER and BA targeted IC, first five phi (0.01-0.05), 1 rep each (as top-degree targeted is deterministic)
# scen=2 for ER targeted, scen=4 for BA targeted
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

def get_targeted_ic(degree, num_I):
    idx = np.argsort(-degree)
    return idx[:num_I]

N = 500
phi_list = [0.01, 0.02, 0.03, 0.04, 0.05]
num_I = 5
results_dir = os.path.join(os.getcwd(),'output')
os.makedirs(results_dir, exist_ok=True)

# ER targeted
scen = 2
adj_path = '/Users/hosseinsamaei/phd/epidemiqs/output/er-adjacency.npz'
adj = scipy.sparse.load_npz(adj_path)
adj = csr_matrix(adj)
degree = np.array(adj.sum(axis=1)).flatten()
stored_result_path = {}
Plot_path = {}
run_details = []
for j, phi in enumerate(phi_list,1):
    initial_I_idx = get_targeted_ic(degree, num_I)
    Slist, Ilist, steps = si_threshold_sim(adj, phi, initial_I_idx, N=N, max_steps=100)
    df = pd.DataFrame({'step': np.arange(len(Slist)), 'S': Slist, 'I': Ilist})
    csv_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}0.csv')
    df.to_csv(csv_path, index=False)
    stored_result_path[csv_path] = f"ER, targeted IC, φ={phi:.2f}, rep 1. Final size: {Ilist[-1]}/500"
    plt.figure()
    plt.plot(df['step'], df['I']/N, label='Defaulted (I)')
    plt.plot(df['step'], df['S']/N, label='Susceptible (S)')
    plt.xlabel('Step')
    plt.ylabel('Fraction')
    plt.title(f'ER, targeted, φ={phi:.2f}, rep 1')
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}0.png')
    plt.savefig(fig_path)
    plt.close()
    Plot_path[fig_path] = f"ER targeted, φ={phi:.2f}, rep 1, time series"
    run_details.append(f"ER, targeted IC, φ={phi:.2f}, rep=1: final I = {Ilist[-1]}/500, steps = {len(df)-1}")
return_vars = ['run_details','stored_result_path','Plot_path']
# BA targeted, same as above
def si_threshold_sim(adjacency, phi, initial_I_idx, N=500, max_steps=100):
    from scipy.sparse import csr_matrix
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

def get_targeted_ic(degree, num_I):
    idx = np.argsort(-degree)
    return idx[:num_I]

import os
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib.pyplot as plt
N = 500
phi_list = [0.01, 0.02, 0.03, 0.04, 0.05]
num_I = 5
results_dir = os.path.join(os.getcwd(),'output')
os.makedirs(results_dir, exist_ok=True)
scen = 4
adj_path = '/Users/hosseinsamaei/phd/epidemiqs/output/ba-adjacency.npz'
adj = scipy.sparse.load_npz(adj_path)
from scipy.sparse import csr_matrix
degree = np.array(adj.sum(axis=1)).flatten()
adj = csr_matrix(adj)
stored_result_path = {}
Plot_path = {}
run_details = []
for j, phi in enumerate(phi_list,1):
    initial_I_idx = get_targeted_ic(degree, num_I)
    Slist, Ilist, steps = si_threshold_sim(adj, phi, initial_I_idx, N=N, max_steps=100)
    df = pd.DataFrame({'step': np.arange(len(Slist)), 'S': Slist, 'I': Ilist})
    csv_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}0.csv')
    df.to_csv(csv_path, index=False)
    stored_result_path[csv_path] = f"BA, targeted IC, φ={phi:.2f}, rep 1. Final size: {Ilist[-1]}/500"
    plt.figure()
    plt.plot(df['step'], df['I']/N, label='Defaulted (I)')
    plt.plot(df['step'], df['S']/N, label='Susceptible (S)')
    plt.xlabel('Step')
    plt.ylabel('Fraction')
    plt.title(f'BA, targeted, φ={phi:.2f}, rep 1')
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(os.getcwd(),'output',f'results-{scen}{j}0.png')
    plt.savefig(fig_path)
    plt.close()
    Plot_path[fig_path] = f"BA targeted, φ={phi:.2f}, rep 1, time series"
    run_details.append(f"BA, targeted IC, φ={phi:.2f}, rep=1: final I = {Ilist[-1]}/500, steps = {len(df)-1}")
return_vars = ['run_details','stored_result_path','Plot_path']