
import pandas as pd
import matplotlib.pyplot as plt
import os
# Static network
df_static = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-10.csv'))
# Temporal network
df_temp = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))
fig, ax = plt.subplots(figsize=(11,6))
ax.plot(df_static['time'], df_static['I'], label='Static Aggregated: Infected', c='r', ls='-')
ax.plot(df_temp['time'], df_temp['I'], label='Temporal (Activity-Driven): Infected', c='b', ls='--')
ax.set_xlabel('Time')
ax.set_ylabel('Number Infectious')
ax.set_title('Comparison of SIR Infection Dynamics\nStatic Aggregated vs Activity-Driven Temporal Network')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'comparison_I.png'))
plt.close()
# Save side-by-side S R curves too for Appendix
fig, axs = plt.subplots(1, 2, figsize=(14,6))
axs[0].plot(df_static['time'], df_static['S'], label='Static: S', c='g')
axs[0].plot(df_static['time'], df_static['I'], label='Static: I', c='r')
axs[0].plot(df_static['time'], df_static['R'], label='Static: R', c='k')
axs[0].set_title('Static Aggregated')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Population')
axs[0].legend()
axs[1].plot(df_temp['time'], df_temp['S'], label='Temporal: S', c='g', ls='--')
axs[1].plot(df_temp['time'], df_temp['I'], label='Temporal: I', c='b', ls='--')
axs[1].plot(df_temp['time'], df_temp['R'], label='Temporal: R', c='k', ls='--')
axs[1].set_title('Activity-Driven Temporal')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Population')
axs[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'comparison_SIR.png'))
plt.close()