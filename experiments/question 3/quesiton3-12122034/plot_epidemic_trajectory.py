
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(times, I, label='Mean Infected (I)', color='red')
plt.fill_between(times, data['SIR_I_90ci_lower'], data['SIR_I_90ci_upper'], color='red', alpha=0.3, label='90% CI Infected')
plt.axvline(x=time_to_peak, color='blue', linestyle='--', label='Time to peak')
plt.axhline(y=peak_prevalence, color='green', linestyle='--', label='Peak prevalence')
plt.xlabel('Time (simulation units)')
plt.ylabel('Number of Individuals')
plt.title('Epidemic Trajectory: Infected over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('output/epidemic_trajectory.png')