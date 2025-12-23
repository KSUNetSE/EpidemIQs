
import networkx as nx
import numpy as np

# We simulate an epidemic on a static network (ER) as a proxy for a time-aggregated network.
# The static network is built with an average degree set to the instantaneous average from the activity-driven model.
# Here, average degree = 2*m*alpha, with m=2 and alpha=0.1, hence avg_degree = 0.4.


def simulate_SIR(N, avg_degree, beta, gamma, seed_node, num_runs=500, random_seed=42):
    np.random.seed(random_seed)
    results = []
    p = avg_degree / (N - 1)  # probability for ER
    for run in range(num_runs):
        G = nx.erdos_renyi_graph(N, p, seed=np.random.randint(0, 10000))
        # S:0, I:1, R:2
        status = np.zeros(N, dtype=int)
        status[seed_node] = 1  # seed infection
        t = 0
        while np.any(status == 1):
            new_status = status.copy()
            infected_nodes = np.where(status == 1)[0]
            for i in infected_nodes:
                for nb in list(G.neighbors(i)):
                    if status[nb] == 0:
                        # Transmission probability per edge, using weight 1/gamma
                        if np.random.rand() < beta / gamma:
                            new_status[nb] = 1
                # Recovery step
                if np.random.rand() < gamma:
                    new_status[i] = 2
            status = new_status
            t += 1
            if t > 10000:
                break
        final_size = np.sum(status == 2) / N
        results.append(final_size)
    return np.array(results)

# Setup parameters from our activity-driven network
N = 1000
m = 2
alpha = 0.1
avg_degree = 2 * m * alpha   # instantaneous average degree is 0.4
beta = 7.5
gamma = 1

# For high activity node simulation: we build one reference network to identify a high degree node.
G_ref = nx.erdos_renyi_graph(N, avg_degree/(N-1))
degrees = np.array([d for _, d in G_ref.degree()])
high_activity_node = int(np.argmax(degrees))

# Simulation runs for high-activity seeding
runs = 200
results_high = simulate_SIR(N, avg_degree, beta, gamma, high_activity_node, num_runs=runs, random_seed=1)

# Simulation runs for random seeding
results_random = []
rng = np.random.default_rng(2)
for i in range(runs):
    random_seed = int(rng.integers(0, 10000))
    random_node = int(rng.integers(0, N))
    res = simulate_SIR(N, avg_degree, beta, gamma, random_node, num_runs=1, random_seed=random_seed)
    results_random.append(res[0])
results_random = np.array(results_random)

# Analysis: outbreak probability (e.g., final epidemic size > threshold) and average final size conditional on outbreak

def analyze_outcomes(results):
    outbreak_threshold = 0.1
    outbreak = results >= outbreak_threshold
    outbreak_probability = outbreak.mean()
    final_sizes = results[outbreak]
    avg_final_size = final_sizes.mean() if len(final_sizes) > 0 else 0
    return outbreak_probability, avg_final_size

outbreak_high, avg_final_high = analyze_outcomes(results_high)
outbreak_random, avg_final_random = analyze_outcomes(results_random)

# Return computed results

results_dict = {
    'outbreak_high': outbreak_high,
    'avg_final_high': avg_final_high,
    'outbreak_random': outbreak_random,
    'avg_final_random': avg_final_random
}

results_dict