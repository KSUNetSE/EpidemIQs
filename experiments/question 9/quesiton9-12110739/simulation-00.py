
# Reasoning and plan
# Step 1: Check that all needed network data file exist, are loadable, and the shape is as expected (500x500 adjacency, exposures)
# Step 2: Load the ER and BA adjacency and exposure matrices, with error handling.
# Step 3: Check degree distribution, plot histograms (validate network structure for context)
# This script will only check/load data for now.
import os
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

current_directory = os.getcwd()
adj_paths = [
    '/Users/hosseinsamaei/phd/epidemiqs/output/er-adjacency.npz',
    '/Users/hosseinsamaei/phd/epidemiqs/output/ba-adjacency.npz',
]
exp_paths = [
    '/Users/hosseinsamaei/phd/epidemiqs/output/er-exposures.npz',
    '/Users/hosseinsamaei/phd/epidemiqs/output/ba-exposures.npz',
]
loaded = {}
err = False
networks_info = {}
for net, exp, label in zip(adj_paths, exp_paths, ['ER', 'BA']):
    try:
        adj = scipy.sparse.load_npz(net)
        expmat = scipy.sparse.load_npz(exp)
        adj = adj.tocsr()
        expmat = expmat.tocsr()
        # Validate adjacency only has 0/1 values
        assert np.all((adj.data == 0) | (adj.data == 1)), f"Adjacency matrix for {label} has non-binary values"
        # Validate size
        assert adj.shape == (500, 500), f"Adjacency wrong shape for {label}: {adj.shape}"
        assert expmat.shape == (500, 500), f"Exposure wrong shape for {label}: {expmat.shape}"
        degree = np.array(adj.sum(axis=1)).flatten()
        networks_info[label] = {
            "adj": adj, "exp": expmat, "degree": degree,
            "avg_deg": float(degree.mean()), "max_deg": int(degree.max()), "min_deg": int(degree.min())
        }
        loaded[label] = True
        # Plot degree dist for sanity
        plt.figure(figsize=(4,2.5))
        plt.hist(degree, bins=min(50,len(np.unique(degree))), alpha=0.7)
        plt.title(f"Degree distribution {label}")
        plt.xlabel('Degree')
        plt.ylabel('#nodes')
        plt.tight_layout()
        plt.savefig(os.path.join(current_directory,'output',f'results-checknet-{label}.png'))
        plt.close()
    except Exception as e:
        loaded[label] = False
        err = True
        print(f"Could not load {label} network: {e}")
        continue
return_vars = ['loaded', 'err', 'networks_info']