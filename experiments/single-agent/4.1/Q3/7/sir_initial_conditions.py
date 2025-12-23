
import numpy as np
import random
N = 1000
initial_infected = 10
# Infect 10 random nodes, rest are susceptible
I_nodes = random.sample(range(N), initial_infected)
S_nodes = [i for i in range(N) if i not in I_nodes]
X0 = np.zeros(N, dtype=int)
X0[I_nodes] = 1 # 0=S, 1=I, 2=R
# Percentage for FastGEMF "percentage" init:
initial_condition_percentage = {"S": (N - initial_infected) * 100 // N, "I": initial_infected * 100 // N, "R": 0}
# Save for simulation reproducibility
import os
np.save(os.path.join(os.getcwd(), "output", "X0_10infected.npy"), X0)
initial_condition_percentage, X0.tolist()