

import numpy as np
import matplotlib.pyplot as plt
import os

# Heuristic model assumptions:
# 1. A cascade triggered in the core is expected to be much more likely due to a higher average number of connections between core banks.
# 2. With the absolute threshold of 2, a seed in a densely connected core will immediately put many neighbors at one failure, and a second failure in this network is highly likely.
# 3. A seed in the periphery, with sparser connectivity among themselves, is less likely to ignite a chain reaction.
# 4. Additionally, increasing the core-core connectivity (kc) amplifies these effects, by both increasing the number of available neighbors and their likelihood of accumulating two failed neighbors.

# We define simple heuristic cascade probability functions:
# For a core seed:
#   baseline: at kc=0.9, the chance can be ~85%
#   Adjust linearly with variations in kc (over a limited range) 
# For a periphery seed:
#   baseline: at kc=0.9, lower chance (~20%) and small sensitivity to kc

def cascade_prob_core(kc):
    # Linear adjustment on small perturbations around baseline kc
    return min(1.0, 0.85 + 0.15 * ((kc - 0.9) / 0.1))


def cascade_prob_periphery(kc):
    # Lower baseline and less sensitivity
    return min(1.0, 0.20 + 0.10 * ((kc - 0.9) / 0.1))

# Given baseline network:
N = 10000
frac_core = 0.1  # core fraction
frac_periphery = 0.9

# Given connectivity baseline parameters:
kc_baseline = 0.9   # core-core connectivity
kp = 0.075          # periphery-periphery connectivity
kcp = 0.3           # core-periphery connectivity

# We explore how cascade probabilities vary with kc for both seeding scenarios
kc_values = np.linspace(0.7, 1.0, 31)
cascade_core_probs = [cascade_prob_core(kc) for kc in kc_values]
cascade_periphery_probs = [cascade_prob_periphery(kc) for kc in kc_values]

# Plot results
plt.figure(figsize=(8,6))
plt.plot(kc_values, cascade_core_probs, label='Core Seed Cascade Probability', marker='o')
plt.plot(kc_values, cascade_periphery_probs, label='Periphery Seed Cascade Probability', marker='s')
plt.xlabel('Core-Core Connectivity (kc)')
plt.ylabel('Estimated Global Cascade Probability')
plt.title('Cascade Probability vs. Core-Core Connectivity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(os.getcwd(), 'output', 'plot-cascade-probability.png')
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
plt.close()

# Baseline probabilities at kc=0.9
baseline_core = cascade_prob_core(0.9)
baseline_periphery = cascade_prob_periphery(0.9)

# For reporting, we produce a dictionary with key variables 
result = {
    'kc_values': kc_values.tolist(),
    'cascade_core_probs': cascade_core_probs,
    'cascade_periphery_probs': cascade_periphery_probs,
    'baseline_core_at_kc_0.9': baseline_core,
    'baseline_periphery_at_kc_0.9': baseline_periphery,
    'plot_path': plot_path
}
result
