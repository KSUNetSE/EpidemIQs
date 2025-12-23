
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
np.random.seed(42)

# Parameters
N = 1000
meank_target = 8  # desired mean degree
powerlaw_exponent = 2  # scale-free exponent
dkmin = 2  # minimum degree to avoid isolated nodes
dkmax = int(N/5)  # set a reasonable upper bound (rare for powerlaw but avoids excessive hubs)

# Generate a power-law degree sequence
# Use the inverse transform sampling method for discrete power-law
def gen_powerlaw_deg_seq(N, gamma, min_k, max_k, meank_target, max_iter=1000):
    for _ in range(max_iter):
        x = np.random.uniform(0, 1, size=N)
        # Compute degree from CDF_inv(x)
        degrees = ((max_k**(1-gamma) - min_k**(1-gamma))*x + min_k**(1-gamma))**(1/(1-gamma))
        degrees = np.floor(degrees).astype(int)
        degrees = np.clip(degrees, min_k, max_k)
        if sum(degrees) % 2 != 0:
            degrees[np.argmax(degrees)] += 1  # force sequence to be graphical
        if np.abs(degrees.mean() - meank_target) < 0.2:
            return degrees
    raise ValueError('Failed to generate graphical power-law degree sequence with target mean.')

powerlaw_degrees = gen_powerlaw_deg_seq(N, powerlaw_exponent, dkmin, dkmax, meank_target)

# Construct configuration model, remove parallel edges and selfloops for realism
g_conf = nx.configuration_model(powerlaw_degrees)
g_conf = nx.Graph(g_conf)  # remove multiedges
g_conf.remove_edges_from(nx.selfloop_edges(g_conf))

fn = os.path.join(os.getcwd(), "output", "scaleffreenetwork.npz")
sparse.save_npz(fn, nx.to_scipy_sparse_array(g_conf))

degrees_conf = np.array([d for n, d in g_conf.degree()])
meank_conf = degrees_conf.mean()
meank2_conf = (degrees_conf**2).mean()

# Plot degree histogram and log-log CCDF
plt.figure(figsize=(6,4))
plt.hist(degrees_conf, bins=np.arange(dkmin, degrees_conf.max()+2)-0.5, color='red', alpha=0.75, rwidth=0.85, log=True)
plt.xlabel("Degree")
plt.ylabel("Count (log scale)")
plt.title("Degree Distribution: Scale-Free Network")
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "output", "degreehistogram-scalefree.png"))
plt.close()

# Log-log degree CCDF
plt.figure(figsize=(6,4))
sorted_degrees = np.sort(degrees_conf)
ccdf = 1.0 - np.arange(len(sorted_degrees)) / float(len(sorted_degrees))
plt.loglog(sorted_degrees, ccdf, marker='o', linestyle='none', color='darkred')
plt.xlabel('Degree')
plt.ylabel('CCDF')
plt.title('Degree CCDF: Scale-Free Network')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "output", "degreeccdf-scalefree.png"))
plt.close()

conf_paths = [fn]
conf_plots = {
    "output/degreehistogram-scalefree.png": "Degree distribution (log scale) of scale-free network",
    "output/degreeccdf-scalefree.png": "Degree CCDF on log-log scale for scale-free network"
}
conf_degree_stats = (meank_conf, meank2_conf)
