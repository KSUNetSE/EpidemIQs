
import pandas as pd
import matplotlib.pyplot as plt
results = pd.read_csv('output/results-11.csv')
plt.figure(figsize=(7,5))
plt.plot(results['time'], results['S'], label='Susceptible (S)')
plt.plot(results['time'], results['I'], label='Infected (I)')
plt.plot(results['time'], results['R'], label='Recovered (R)')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('SIR Dynamics over Barabási–Albert Network')
plt.legend()
plt.tight_layout()
plt.savefig('output/sir_summarized_11.png')
plt.close()
# Describe the summarized shape and features numerically
s_min = results['S'].min()
s_return = results['S'].iloc[-1] - s_min
r_final = results['R'].iloc[-1]
i_max = results['I'].max()
i_max_time = results['time'][results['I'].idxmax()]
desc = f"Susceptibles decline sharply then plateau, minimal return ({s_return} individuals). Recovered increases to final epidemic size {r_final}. Infected peaks at {i_max} at time {i_max_time:.2f} days, then declines. Total epidemic lasts about {results['time'][results['I']>0].iloc[-1]:.1f} days."
desc