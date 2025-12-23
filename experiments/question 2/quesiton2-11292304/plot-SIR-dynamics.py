
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(data['time'], data['S'], label='Mean Susceptible')
plt.fill_between(data['time'], data['S_90ci_lower'], data['S_90ci_upper'], color='g', alpha=0.2, label='S 90% CI')
plt.plot(data['time'], data['I'], label='Mean Infected', color='orange')
plt.fill_between(data['time'], data['I_90ci_lower'], data['I_90ci_upper'], color='orange', alpha=0.2, label='I 90% CI')
plt.plot(data['time'], data['R'], label='Mean Recovered', color='red')
plt.fill_between(data['time'], data['R_90ci_lower'], data['R_90ci_upper'], color='red', alpha=0.2, label='R 90% CI')
plt.axvline(x=t_peak_mean, color='purple', linestyle='--', label='Mean t_peak')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Dynamics on Scale-Free Network (BA) - Scenario 4')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/SIR_dynamics_scenario4.png')
plt.close()