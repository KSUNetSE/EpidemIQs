
# The error was due to appending tuples (counts) in time_series as one item per step, then trying to convert to np.array: this gives an array of tuples.
# Fix: use separate lists for S and I; or make time_series as shape (step,2) array directly.
# Test the revised logic for the small network ring cascade.
import numpy as np
import scipy.sparse

def si_threshold_sim(adjacency, phi, initial_I_idx, N=10, max_steps=20):
    adjacency = adjacency.tocsr()
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
            if k == 0:
                continue
            neighbor_idxs = adjacency[i].indices
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

# Small test: ring network 10 nodes, phi=0.5, initial I = [0]
A = np.zeros((10,10),dtype=int)
for i in range(10):
    A[i,(i+1)%10] = 1
    A[i,(i-1)%10] = 1
adj = scipy.sparse.csr_matrix(A)
initial_I = [0]
phi = 0.5
Slist, Ilist, steps = si_threshold_sim(adj, phi, initial_I, N=10, max_steps=20)
return_vars = ['Slist','Ilist','steps']