
# Chain-of-Thought:
# Next, construct SIR model schema for fastgemf for both static and temporal networks. Model compartments: [S, I, R].
# For initial condition: let 1% infected (I), 99% susceptible (S), 0% recovered (R), random assignment.
# Prepare initial condition dict and save.
import numpy as np
N = 1000
init_S = int(0.99 * N)
init_I = N - init_S  # 1% infected
init_R = 0
init_conditions = {'percentage': {'S': int(init_S*100/N), 'I': int(init_I*100/N), 'R': int(init_R*100/N)}}
# Save to file as well (for reproducibility)
import json, os
with open(os.path.join(os.getcwd(), 'output', 'initial_condition.json'), 'w') as f:
    json.dump(init_conditions, f)
init_conditions
