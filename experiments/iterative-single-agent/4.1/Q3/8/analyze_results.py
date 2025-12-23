
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load static simulation data
csv_static = os.path.join('output','results-11.csv')
df_static = pd.read_csv(csv_static)

# Load temporal simulation data
csv_temp = os.path.join('output','results-12.csv')
df_temp = pd.read_csv(csv_temp)

# Compute relevant metrics:
def epidemic_metrics(df):
    total_infected = df['R'].iloc[-1]
    peak_I = df['I'].max()
    t_peak = df['I'].idxmax()
    dur = df['time'].iloc[-1] if df['I'].iloc[-1]==0 else df['time'][df['I']>0].iloc[-1]
    final_outbreak = total_infected/1000.0
    return {'FinalSize': total_infected, 'PeakI': peak_I, 'TimeToPeak': df['time'][t_peak], 'Duration': dur, 'AttackRate': final_outbreak}

metrics_static = epidemic_metrics(df_static)
metrics_temp = epidemic_metrics(df_temp)

# Plot overlays
plt.figure(figsize=(8,5))
plt.plot(df_static['time'], df_static['I'], label='Static aggregated: I(t)', color='tab:red')
plt.plot(df_temp['time'], df_temp['I'], label='Temporal ADN: I(t)', color='tab:blue')
plt.xlabel('Time')
plt.ylabel('Number Infectious')
plt.legend()
plt.title('SIR Infection Curves: Static vs Temporal Network')
plt.tight_layout()
plt.savefig(os.path.join('output','compare_It.png'))

# Also plot R(t)
plt.figure()
plt.plot(df_static['time'], df_static['R'], label='Static aggregated: R(t)', color='tab:orange')
plt.plot(df_temp['time'], df_temp['R'], label='Temporal ADN: R(t)', color='tab:green')
plt.xlabel('Time')
plt.ylabel('Cumulative Recovered')
plt.legend()
plt.title('SIR Recovered Curves: Static vs Temporal Network')
plt.tight_layout()
plt.savefig(os.path.join('output','compare_Rt.png'))

# Save summary metrics
summary_table = pd.DataFrame([
    {**{'Model':'Static'}, **metrics_static},
    {**{'Model':'Temporal'}, **metrics_temp},
])
summary_table.to_csv(os.path.join('output','summary_metrics.csv'), index=False)
