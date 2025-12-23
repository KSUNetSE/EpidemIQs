
# Manually analyze plots to compare infection peaks, timing, tail, other differences.
import matplotlib.pyplot as plt
import pandas as pd

# 1. Stochastic ER
sim_er = pd.read_csv('output/results-11.csv')
# 2. Stochastic BA
sim_ba = pd.read_csv('output/results-12.csv')
# 3. Deterministic ODE
sim_det = pd.read_csv('output/results-13.csv')

plt.figure(figsize=(8,6))
plt.plot(sim_er['time'], sim_er['I'], label='Stochastic ER', color='b', lw=2)
plt.plot(sim_ba['time'], sim_ba['I'], label='Stochastic BA', color='orange', lw=2)
plt.plot(sim_det['time'], sim_det['I'], label='Mean-field ODE', color='g', lw=2)
plt.xlabel('Time (days)')
plt.ylabel('Infected Population, I')
plt.legend()
plt.title('Comparison of Infection Peaks: Homogeneous vs Degree-Heterogeneous SEIR')
plt.tight_layout()
plt.savefig('output/seir_infection_comparison.png')
plt.close()

# Summarize key point differences
results_summary = {
    'peak_I': {
        'ER': float(sim_er['I'].max()),
        'BA': float(sim_ba['I'].max()),
        'ODE': float(sim_det['I'].max())
    },
    'time_peak': {
        'ER': float(sim_er['time'][sim_er['I'].idxmax()]),
        'BA': float(sim_ba['time'][sim_ba['I'].idxmax()]),
        'ODE': float(sim_det['time'][sim_det['I'].idxmax()])
    },
    'final_size': {
        'ER': float(sim_er['R'].iloc[-1]),
        'BA': float(sim_ba['R'].iloc[-1]),
        'ODE': float(sim_det['R'].iloc[-1])
    },
    'tail_features': {
        'ER': float(sim_er[sim_er['I'] < 1]['time'].iloc[0]) if (sim_er['I']<1).any() else None,
        'BA': float(sim_ba[sim_ba['I'] < 1]['time'].iloc[0]) if (sim_ba['I']<1).any() else None,
        'ODE': float(sim_det[sim_det['I'] < 1]['time'].iloc[0]) if (sim_det['I']<1).any() else None
    }
}
print(results_summary)
