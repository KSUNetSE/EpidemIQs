
# Fix population sequence bug: use sorted(G.nodes)
import networkx as nx
import numpy as np
import pandas as pd
import os
import random

output_dir = os.path.join(os.getcwd(), 'output')

g_agg = nx.erdos_renyi_graph(1000, 39.81/999)
g_low = nx.erdos_renyi_graph(1000, 0.448/999)

def run_SIR(G, beta, gamma, t_max=60, initial_infected_frac=0.01):
    N = G.number_of_nodes()
    S = set(G.nodes())
    node_sequence = sorted(G.nodes())
    I = set(random.sample(node_sequence, int(initial_infected_frac*N)))
    S -= I
    R = set()
    times, Ss, Is, Rs = [], [], [], []
    for t in range(t_max):
        new_I = set()
        new_R = set()
        for node in I:
            for nbr in G.neighbors(node):
                if nbr in S and random.uniform(0,1)<beta:
                    new_I.add(nbr)
            if random.uniform(0,1)<gamma:
                new_R.add(node)
        S -= new_I
        I = (I | new_I) - new_R
        R |= new_R
        times.append(t)
        Ss.append(len(S))
        Is.append(len(I))
        Rs.append(len(R))
        if not I:
            break
    return pd.DataFrame({'time': times, 'S': Ss, 'I': Is, 'R': Rs})

beta_agg = 0.07358  # from earlier calculation
beta_low = 0.1405

gamma = 0.2

# Rerun with fixed bug
df_agg = run_SIR(g_agg, beta_agg, gamma, t_max=60)
df_low = run_SIR(g_low, beta_low, gamma, t_max=300)

agg_csv = os.path.join(output_dir, 'results-1-1.csv')
low_csv = os.path.join(output_dir, 'results-1-2.csv')
df_agg.to_csv(agg_csv, index=False)
df_low.to_csv(low_csv, index=False)

agg_csv, low_csv