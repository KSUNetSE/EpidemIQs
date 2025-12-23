
import matplotlib.pyplot as plt
import os

# Set up plot
fig, ax = plt.subplots(figsize=(10,6))

# Plot mean I(t) curve
ax.plot(data['time'], data['I'], label='Mean Infected (I)', color='blue')

# Fill 90% confidence interval
ax.fill_between(data['time'], data['I_90ci_lower'], data['I_90ci_upper'], color='blue', alpha=0.2, label='90% CI')

# Mark key points: start (time=0), peak, end (epidemic duration)
ax.plot(0, data.loc[data['time']==0, 'I'].values[0], 'go', label='Start I')
ax.plot(peak_time, peak_I, 'ro', label='Peak Infection')
ax.plot(epidemic_duration, 0, 'ko', label='Epidemic End')

# Annotate points
ax.annotate(f'Peak: {peak_I:.1f} ({peak_I_pct:.1f}%)\nDay: {peak_time}', xy=(peak_time, peak_I), xytext=(peak_time+2, peak_I+10), arrowprops=dict(facecolor='red', shrink=0.05))
ax.annotate(f'End day: {epidemic_duration}', xy=(epidemic_duration, 0), xytext=(epidemic_duration-10, 10), arrowprops=dict(facecolor='black', shrink=0.05))

# Labels and legend
ax.set_xlabel('Time (days)')
ax.set_ylabel('Number of Infected Individuals')
ax.set_title('SIR Epidemic Simulation: Infected Population over Time')
ax.legend()
ax.grid(True)

# Save plot
output_dir = os.path.join(os.getcwd(), 'output')
plot_path = os.path.join(output_dir, 'sir_simulation_infected_curve.png')
fig.savefig(plot_path)
plt.close(fig)

plot_path