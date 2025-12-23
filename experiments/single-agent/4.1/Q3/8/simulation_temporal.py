
import fastgemf as fg
import numpy as np
import scipy.sparse as sparse
import os

beta_temp = 0.7514 # from parameter setup
recovery_rate = 0.2
N = 1000

# For the temporal simulation, we'll re-generate activity-driven snapshots 'on the fly' for each step
# We'll run a simplified version for a proof-of-concept

def run_temporal_sir_sim(N, alpha, m, beta, gamma, T, init_frac_I=0.01, nrep=1):
    # 0: S, 1: I, 2: R
    results = {'time': [], 'S': [], 'I': [], 'R': []}
    for rep in range(nrep):
        X = np.zeros(N, dtype=int)
        infected0 = np.random.choice(N, size=int(N*init_frac_I), replace=False)
        X[infected0] = 1
        t = 0.0
        S, I, R = [np.sum(X==0)], [np.sum(X==1)], [np.sum(X==2)]
        times = [t]
        for step in range(T):
            active = np.random.rand(N) < alpha
            events = []
            # List all possible S-I contacts this round
            for i in np.where(active)[0]:
                targets = np.random.choice(np.delete(np.arange(N), i), size=m, replace=False)
                for j in targets:
                    # S-I pair
                    if (X[i]==1 and X[j]==0) or (X[i]==0 and X[j]==1):
                        events.append( (i,j) )
            # Apply infections
            new_inf = set()
            for (i,j) in events:
                if X[i]==1 and X[j]==0 and np.random.rand() < beta:
                    new_inf.add(j)
                if X[i]==0 and X[j]==1 and np.random.rand() < beta:
                    new_inf.add(i)
            for idx in new_inf:
                X[idx]=1
            # Recoveries
            infected = np.where(X==1)[0]
            recoveries = infected[np.random.rand(len(infected)) < gamma]
            X[recoveries]=2
            S.append(np.sum(X==0)); I.append(np.sum(X==1)); R.append(np.sum(X==2))
            t += 1.0
            times.append(t)
        # Store only 1st rep for csv
        if rep==0:
            results['time'], results['S'], results['I'], results['R'] = times, S, I, R
    return results

# Run the simulation for T=100
results = run_temporal_sir_sim(N=1000, alpha=0.1, m=2, beta=0.7514, gamma=0.2, T=100, init_frac_I=0.01, nrep=1)
import pandas as pd
data = pd.DataFrame(results)
data.to_csv(os.path.join(os.getcwd(),'output','results-12.csv'),index=False)
