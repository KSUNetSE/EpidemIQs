
import matplotlib.pyplot as plt
import os

# Summary metrics (mean and 90% CI span)
summary_metrics = {
    'Metric': [
        'Estimated I_peak (mean)', 
        'Time of I_peak (mean)',
        'Final Susceptible (S_inf)', 
        'Final Recovered (epidemic size)',
        'Duration (time units)'
    ],
    'Value': [I_peak, t_peak, S_final, R_final, duration],
    'Unit': ['individuals', 'time units', 'individuals', 'individuals', 'time units']
}

summary_df = pd.DataFrame(summary_metrics)

# Plot trajectories of S, I, R with 90% CI
plt.figure(figsize=(10,6))
plt.plot(data['time'], data['S'], label='S (mean)', color='blue')
plt.fill_between(data['time'], data['S_90ci_lower'], data['S_90ci_upper'], color='blue', alpha=0.3)
plt.plot(data['time'], data['I'], label='I (mean)', color='red')
plt.fill_between(data['time'], data['I_90ci_lower'], data['I_90ci_upper'], color='red', alpha=0.3)
plt.plot(data['time'], data['R'], label='R (mean)', color='green')
plt.fill_between(data['time'], data['R_90ci_lower'], data['R_90ci_upper'], color='green', alpha=0.3)

plt.xlabel('Time (time units)')
plt.ylabel('Individuals')
plt.title('Average SIR trajectories with 90% CI')
plt.legend()
plt.grid(True)

# Ensure output directory exists
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# Save plot
plot_path = os.path.join(output_dir, 'sir_avg_trajectories.png')
plt.savefig(plot_path)

summary_df, plot_path