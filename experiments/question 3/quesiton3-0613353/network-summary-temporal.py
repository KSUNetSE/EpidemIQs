
# For documentation: get node/edge counts, mean degree/strength for the temporal network as well
import numpy as np
import pandas as pd
import os

data_dir = os.path.join(os.getcwd(),'output')
activity = np.load(os.path.join(data_dir,'exp3-powerlaw-activity.npy'))
edge_df = pd.read_csv(os.path.join(data_dir,'exp3-temporal-contact-edges.csv'))
N = activity.shape[0]
Tmax = edge_df['t'].max() + 1
# Per-step contacts
contacts_per_step = edge_df.groupby('t').size()
# All unique node pairs
all_edges = set(tuple(sorted((i,j))) for i,j in zip(edge_df['i'],edge_df['j']))
per_step_mean_degree = contacts_per_step.mean() * 2 / N
network_stats = {
    'N': N,
    'timesteps': Tmax,
    'unique_edges': len(all_edges),
    'per_step_contacts_mean': float(contacts_per_step.mean()),
    'per_step_contacts_std': float(contacts_per_step.std()),
    'per_step_mean_degree': float(per_step_mean_degree)
}