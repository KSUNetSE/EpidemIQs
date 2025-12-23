
# Synthesize modeling for the temporal activity-driven network using FastGEMF-like simulation for one-step at a time (approximation)
# Since we do not have a built-in activity-driven simulation in this context, we approximate by emulating the activation rule
import numpy as np
import pandas as pd
import random
import os

N = 1000
alpha = 0.1
m = 2
t_max = 300
beta = 0.15  # Set slightly higher than low mean-degree to test effect
gamma = 0.2
init_inf_frac = 0.01
output_dir = os.path.join(os.getcwd(), 'output')

# Initial states
S = set(range(N))
I = set(random.sample(range(N), int(init_inf_frac * N)))
S -= I
R = set()
times, Ss, Is, Rs = [], [], [], []

for t in range(t_max):
    # For each active node, pick m distinct partners to link to
    links = set()
    active = [node for node in range(N) if random.uniform(0,1) < alpha]
    for node in active:
        partners = random.sample([i for i in range(N) if i != node], m)
        for p in partners:
            links.add((node,p) if node<p else (p,node))
    # Collect infected and susceptible sets
    new_I = set()
    new_R = set()
    for u, v in links:
        if u in I and v in S:
            if random.uniform(0,1)<beta:
                new_I.add(v)
        elif v in I and u in S:
            if random.uniform(0,1)<beta:
                new_I.add(u)
    # Recovery
    for i in list(I):
        if random.uniform(0,1)<gamma:
            new_R.add(i)
    S -= new_I
    I = (I | new_I) - new_R
    R |= new_R
    times.append(t)
    Ss.append(len(S))
    Is.append(len(I))
    Rs.append(len(R))
    if not I:
        break

df = pd.DataFrame({'time': times, 'S': Ss, 'I': Is, 'R': Rs})
csv_path = os.path.join(output_dir, 'results-1-3.csv')
df.to_csv(csv_path, index=False)
csv_path