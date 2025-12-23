
import pandas as pd
import matplotlib.pyplot as plt
import os
output_dir = os.path.join(os.getcwd(), 'output')
df = pd.read_csv(os.path.join(output_dir, 'results-1-3.csv'))
plt.figure(figsize=(7,3))
plt.plot(df['time'], df['I'], label='Temporal Activity-Driven')
plt.xlabel('Time')
plt.ylabel('Infected')
plt.title('SIR: Activity-Driven Temporal Network')
plt.legend()
plt.tight_layout()
fig_path = os.path.join(output_dir, 'sir-results-temporal.png')
plt.savefig(fig_path)
plt.close()
# Compute peak and final size
peak_I = df['I'].max()
peak_time = df['time'][df['I'].idxmax()]
final_R = df['R'].iloc[-1]
metrics = {'peak_infection': int(peak_I), 'peak_time': float(peak_time), 'final_epidemic_size': int(final_R)}
fig_path, metrics