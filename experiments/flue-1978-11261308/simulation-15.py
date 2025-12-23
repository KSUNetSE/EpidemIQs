
# Step 4: Overlay best simulation with empirical B(t) and C(t)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load simulation for beta=0.3789 (results-11.csv)
sim_df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))
emp_df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-14.csv'))

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# Simulated B: mean + CI
ax.plot(sim_df['time'], sim_df['B'], label='Simulated B(t)', color='C0', linewidth=2)
ax.fill_between(sim_df['time'], sim_df['B_lower'], sim_df['B_upper'], color='C0', alpha=0.2, label='Sim B(t) 90% CI')
# Overlay empirical B(t)
ax.scatter(emp_df['day'], emp_df['B_obs'], color='black', marker='o', s=30, label='Empirical B(t) (observed)')
# Sim R (cumulative convalescent)
ax.plot(sim_df['time'], sim_df['R'], label='Simulated Cumulative R (Convalescent)', color='C3', linestyle='--')
# Empirical C curve
ax.plot(emp_df['day'], emp_df['C_obs'], color='gray', linestyle=':', label='Empirical Cumulative R (Convalescent)')

ax.set_xlabel('Days')
ax.set_ylabel('Individuals')
ax.set_title('SEIBR Simulation vs Empirical Data (English Boarding School Flu)')
ax.legend()
plt.tight_layout()
overlay_path = os.path.join(os.getcwd(), 'output', 'results-15.png')
plt.savefig(overlay_path)
plt.close()

return_vars = ['overlay_path']
