
# Load and analyze both CSV files, extract metrics (peak I, time, final size, duration)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_er = pd.read_csv('output/results-1-1.csv')
df_conf = pd.read_csv('output/results-1-2.csv')
# Standard epidemic metrics:
def get_metrics(df, N):
    peak_I = df['I'].max()
    peak_time = df['time'][df['I'].idxmax()]
    final_size = df['R'].iloc[-1]
    epi_duration = df[df['I']>0].shape[0]  # time steps with I>0
    area_I = np.trapz(df['I'], df['time']) # total infectious load
    return {'peak_I': int(peak_I), 'peak_time': float(peak_time), 'final_size': int(final_size), 'duration': int(epi_duration), 'area_I':float(area_I)}
N = 2500
metrics_er = get_metrics(df_er, N)
metrics_conf = get_metrics(df_conf, N)
# Overlay plot for Figure
plt.figure(figsize=(8,5))
plt.plot(df_er['time'], df_er['I'], label='I (ER)', lw=2)
plt.plot(df_conf['time'], df_conf['I'], label='I (Powerlaw)', lw=2)
plt.plot(df_er['time'], df_er['E'], '--', label='E (ER)', lw=1.2)
plt.plot(df_conf['time'], df_conf['E'], '--', label='E (PL)', lw=1.2)
plt.xlabel('Time'); plt.ylabel('Individuals'); plt.title('SEIR: Infected and Exposed (ER vs. PL)')
plt.legend(); plt.tight_layout();
plt.savefig('output/seir-compare.png'); plt.close()
summary_df = pd.DataFrame({'network':['ER (homogeneous)', 'Powerlaw (heterogeneous)'],
   'Peak Infected': [metrics_er['peak_I'], metrics_conf['peak_I']],
   'Time of Peak':[metrics_er['peak_time'], metrics_conf['peak_time']],
   'Final Size (R)': [metrics_er['final_size'], metrics_conf['final_size']],
   'Epidemic Duration':[metrics_er['duration'], metrics_conf['duration']],
   'Total Infectious Load (area I)': [metrics_er['area_I'], metrics_conf['area_I']]})
summary_df.to_csv('output/seir-metrics.csv', index=False)
metrics_er, metrics_conf