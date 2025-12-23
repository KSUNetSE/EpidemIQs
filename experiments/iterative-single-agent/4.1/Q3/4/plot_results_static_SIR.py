
# Chain-of-Thought: Now plot the results from CSV results file for static SIR to produce a compartment evolution PNG. This avoids plotting during simulation, minimizing errors.
import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv(os.path.join(os.getcwd(),'output','results-1-1.csv'))
plt.figure(figsize=(8,5))
plt.plot(df['time'], df['S'], label='Susceptible')
plt.plot(df['time'], df['I'], label='Infected')
plt.plot(df['time'], df['R'], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR dynamics on time-aggregated static network')
plt.legend()
plt.tight_layout()
fig_path = os.path.join(os.getcwd(),'output','results-1-1.png')
plt.savefig(fig_path)
fig_path
