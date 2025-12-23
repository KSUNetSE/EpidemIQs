
import pandas as pd
import matplotlib.pyplot as plt
import os

data = pd.read_csv(os.path.join('output', 'results-11.csv'))
# Basic metrics
N = 200

duration = data['time'].iloc[-1] - data['time'].iloc[0]
final_I1 = data['I1'].iloc[-1]
final_I2 = data['I2'].iloc[-1]
peak_I1 = data['I1'].max()
peak_time_I1 = data['time'][data['I1'].idxmax()]
peak_I2 = data['I2'].max()
peak_time_I2 = data['time'][data['I2'].idxmax()]
# Equilibrium analysis: do both I1, I2 >0 at end?
coexist_equilibrium = int(final_I1 > 0 and final_I2 > 0)
final_state = 'coexist' if coexist_equilibrium else ('I1 dominates' if final_I1 > 0 else 'I2 dominates')
# Plot infection trajectories for inclusion in report
plt.figure(figsize=(7,5))
plt.plot(data['time'], data['I1'], label='Virus 1 (I1)', color='blue')
plt.plot(data['time'], data['I2'], label='Virus 2 (I2)', color='orange')
plt.xlabel('Time')
plt.ylabel('# Infected')
plt.title('Competitive SIS Epidemics: Virus 1 vs Virus 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
results_fig_path = os.path.join('output', 'results-infectious_vs_time.png')
plt.savefig(results_fig_path)
metrics = {
    'Total duration': float(duration),
    'Final I1': int(final_I1),
    'Final I2': int(final_I2),
    'Peak I1': int(peak_I1),
    'Peak time I1': float(peak_time_I1),
    'Peak I2': int(peak_I2),
    'Peak time I2': float(peak_time_I2),
    'Coexistence?': bool(coexist_equilibrium),
    'Final state': final_state
}
metrics
results_fig_path
