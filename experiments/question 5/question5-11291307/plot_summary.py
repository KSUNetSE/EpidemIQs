
import matplotlib.pyplot as plt

# Creating summary plots for both datasets for visual confirmation

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot for results-11
axs[0].plot(results_11['time'], results_11['S'], label='Susceptible')
axs[0].plot(results_11['time'], results_11['I'], label='Infected')
axs[0].plot(results_11['time'], results_11['R'], label='Recovered')
axs[0].set_title('SIR dynamics under random vaccination (75% coverage)')
axs[0].set_ylabel('Number of individuals')
axs[0].legend()

# Plot for results-21
axs[1].plot(results_21['time'], results_21['S'], label='Susceptible')
axs[1].plot(results_21['time'], results_21['I'], label='Infected')
axs[1].plot(results_21['time'], results_21['R'], label='Recovered')
axs[1].set_title('SIR dynamics under targeted vaccination (degree-10 nodes)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Number of individuals')
axs[1].legend()

plt.tight_layout()
plt.savefig('output/sir_simulation_summary.png')