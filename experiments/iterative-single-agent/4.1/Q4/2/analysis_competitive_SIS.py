
import pandas as pd
import matplotlib.pyplot as plt
import os
# Load the simulation results
csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
data = pd.read_csv(csv_path)
# Extract peak infection, final prevalence, and time to peak for both strains
I1 = data['I1']
I2 = data['I2']
t = data['time']
# Metrics
peak_I1 = I1.max()
peak_time_I1 = t[I1.idxmax()]
final_I1 = I1.iloc[-1]
peak_I2 = I2.max()
peak_time_I2 = t[I2.idxmax()]
final_I2 = I2.iloc[-1]
duration = t[I1.gt(0) | I2.gt(0)].max() - t[I1.gt(0) | I2.gt(0)].min()
# Plot dynamics for inclusion in appendix if required
plt.figure()
plt.plot(t, I1, label='Virus 1 (Layer A)', linestyle='-')
plt.plot(t, I2, label='Virus 2 (Layer B)', linestyle='--')
plt.xlabel('Time')
plt.ylabel('No. Infected')
plt.title('Competitive SIS Process')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'competitiveSIS_infection_time.png'))
plt.close()
# Metrics as dict
metrics = {
    'peak_I1': int(peak_I1), 'peak_time_I1': float(peak_time_I1), 'final_I1': int(final_I1),
    'peak_I2': int(peak_I2), 'peak_time_I2': float(peak_time_I2), 'final_I2': int(final_I2),
    'duration': float(duration)
}
