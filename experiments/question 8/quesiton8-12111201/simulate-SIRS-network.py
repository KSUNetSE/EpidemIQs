
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def simulate_SIRS(network, beta, gamma, xi, T, dt=0.1, init_infect_frac=0.05):
    # Model: U, P, F with transitions:
    # U -> P with per contact probability = beta * dt if neighbor in state P
    # P -> F with probability = gamma*dt
    # F -> U with probability = xi*dt
    # states: 0: U, 1: P, 2: F
    n = len(network.nodes())
    states = np.zeros(n, dtype=int)  # initially all U
    # randomly set a number of P
    init_infected = np.random.choice(n, int(init_infect_frac*n), replace=False)
    states[init_infected] = 1
    # time series lists
    U_list, P_list, F_list = [], [], []
    times = []
    steps = int(T/dt)
    nodes = list(network.nodes())
    for step in range(steps):
        new_states = states.copy()
        # For each node, process updates
        for i in nodes:
            if states[i] == 0:  # U
                # contact with neighbors in state P
                # probability to become P is 1 - exp(-beta*#neighbors*dt) approx beta*#neighbors in dt if small
                p_neighbors = sum(states[j] == 1 for j in network.neighbors(i))
                if np.random.rand() < 1 - np.exp(-beta * p_neighbors * dt):
                    new_states[i] = 1
            elif states[i] == 1:  # P
                if np.random.rand() < gamma*dt:
                    new_states[i] = 2
            elif states[i] == 2:  # F
                if np.random.rand() < xi*dt:
                    new_states[i] = 0
        states = new_states
        U_list.append(np.mean(states == 0))
        P_list.append(np.mean(states == 1))
        F_list.append(np.mean(states == 2))
        times.append(step*dt)
    return np.array(times), np.array(U_list), np.array(P_list), np.array(F_list)

# Generate a small-world network using Watts-Strogatz model
N = 1000
k = 10  # each node connected to k neighbours (should be even)
p_rewire = 0.1
network = nx.watts_strogatz_graph(N, k, p_rewire)

# Parameter values: choosing beta, gamma, xi as discussed. Adjust to see impact.
# We'll simulate for T time units
beta = 0.8  # posting/infection rate per day
gamma = 0.3 # fatigue rate per day
xi = 0.1    # forgetting rate per day
T = 160     

times, U_series, P_series, F_series = simulate_SIRS(network, beta, gamma, xi, T, dt=0.1)

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(times, U_series, label='Unaware (U)')
plt.plot(times, P_series, label='Posting (P)')
plt.plot(times, F_series, label='Fatigued (F)')
plt.xlabel('Time')
plt.ylabel('Fraction of population')
plt.title('SIRS Simulation on a Small-World Network')
plt.legend()

# Save plot to specified location
import os
plot_path = os.path.join(os.getcwd(), 'output', 'SIRS-Simulation.png')
if not os.path.exists(os.path.dirname(plot_path)):
    os.makedirs(os.path.dirname(plot_path))
plt.savefig(plot_path)
plt.close()

result = {'times': times, 'U_series': U_series, 'P_series': P_series, 'F_series': F_series, 'plot_path': plot_path}
result