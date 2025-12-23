
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os


def simulate_epidemic(G, initial_infected, T, recovery_prob, max_time=100):
    # SIR simulation on network G. Nodes: susceptible (0), infected (1), recovered (2)
    state = {node: 0 for node in G.nodes()}
    for n in initial_infected:
        state[n] = 1
    infected_nodes = set(initial_infected)
    recovered_nodes = set()
    sizes = []
    t = 0
    while infected_nodes and t < max_time:
        new_infected = set()
        new_recovered = set()
        for node in list(infected_nodes):
            for neighbor in G.neighbors(node):
                if state[neighbor] == 0 and np.random.rand() < T:
                    new_infected.add(neighbor)
            # attempt recovery
            if np.random.rand() < recovery_prob:
                new_recovered.add(node)
        for node in new_infected:
            state[node] = 1
        for node in new_recovered:
            state[node] = 2
        infected_nodes = (infected_nodes.union(new_infected)).difference(new_recovered)
        recovered_nodes = recovered_nodes.union(new_recovered)
        sizes.append(len(recovered_nodes))
        t += 1
    return sizes, state

# To validate the threshold p_c = 0.75 from the analytic derivation, one can follow these simulation steps:
# 1. Generate a network (here an Erdos-Renyi network with n nodes and mean degree z=3) as a proxy for a configuration model network.
# 2. Randomly remove a fraction p of nodes (i.e. vaccination). Removing nodes reduces the size/structure of the network.
# 3. Simulate an SIR epidemic with a high transmissibility T to ensure widespread outbreak in absence of vaccination. Here, we set T=1.0 so that the main driver is network connectivity.
# 4. Record the final epidemic size; running this simulation repeatedly for various p values should show that the epidemic outbreak becomes very limited as p exceeds 0.75.

# For demonstration, we run the simulation over various vaccination fractions.

n = 3000
z = 3
p_values = np.linspace(0, 0.95, 10)

# In the analytic formula, R0 = 4 requires T * q = 4 (with q=4) so T = 1.0 in our simulation. 
T = 1.0 
recovery_prob = 0.2  # chosen arbitrarily; setting T high ensures most epidemic behavior relies on connectivity.

final_sizes = []

for p in p_values:
    # Generate an Erdos-Renyi network
    G = nx.erdos_renyi_graph(n, z/(n-1))
    # Vaccinate p fraction of nodes: remove them
    nodes_to_remove = np.random.choice(list(G.nodes()), int(p*n), replace=False)
    G.remove_nodes_from(nodes_to_remove)
    if len(G) == 0:
        final_sizes.append(0)
        continue
    # Start epidemic with one randomly infected node
    initial_infected = list(np.random.choice(list(G.nodes()), 1))
    sizes, state = simulate_epidemic(G, initial_infected, T, recovery_prob)
    # Record final epidemic size
    final_sizes.append(sizes[-1] if sizes else 0)

# Plot the epidemic final sizes vs vaccination fraction.
plt.figure(figsize=(8,5))
plt.plot(p_values, final_sizes, marker='o', linestyle='-')
plt.xlabel('Vaccination Fraction p')
plt.ylabel('Final Epidemic Size')
plt.title('Epidemic Final Size vs Vaccination Fraction')
plt.grid(True)

# Save the plot to the required path
plot_path = os.path.join(os.getcwd(), 'output', 'plot-epidemic-simulation.png')
# Ensure the output directory exists
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
plt.close()

final_sizes_summary = {'p_values': p_values.tolist(), 'final_sizes': final_sizes, 'plot_path': plot_path}

final_sizes_summary