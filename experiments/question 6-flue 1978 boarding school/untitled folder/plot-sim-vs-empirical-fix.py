
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot Bed-confined (B)
axs[0].plot(empirical_days, empirical_B, label='Empirical B', marker='o', linestyle='-', color='blue')
axs[0].plot(empirical_days, sim_B_interp, label='Simulated B', linestyle='--', color='cyan')
axs[0].set_ylabel('Number of Bed-confined Individuals')
axs[0].set_title('Bed-confined (B) individuals: Simulated vs Empirical')

# Annotate peak and final size for B
axs[0].annotate(f'Peak Empirical B = {emp_peak_B_value} at day {emp_peak_B_time}',
                xy=(emp_peak_B_time, emp_peak_B_value), xytext=(emp_peak_B_time+1, emp_peak_B_value+20),
                arrowprops=dict(facecolor='blue', shrink=0.05), color='blue')
axs[0].annotate(f'Peak Simulated B = {peak_B_value:.1f} at day {peak_B_time:.1f}',
                xy=(peak_B_time, peak_B_value), xytext=(peak_B_time+3, peak_B_value-50),
                arrowprops=dict(facecolor='cyan', shrink=0.05), color='cyan')

# Plot Convalescent (C)
axs[1].plot(empirical_days, empirical_C, label='Empirical C', marker='o', linestyle='-', color='red')
axs[1].plot(empirical_days, sim_C_interp, label='Simulated C', linestyle='--', color='orange')
axs[1].set_ylabel('Number of Convalescent Individuals')
axs[1].set_title('Convalescent (C) individuals: Simulated vs Empirical')
axs[1].set_xlabel('Days from Outbreak Start')

# Annotate epidemic duration window
for ax in axs:
    ax.axvspan(emp_first_day, emp_last_day, color='lightgrey', alpha=0.3, label='Empirical Epidemic Duration')
    ax.axvspan(first_day_corrected, last_day_corrected, color='yellow', alpha=0.3, label='Simulated Epidemic Duration')
    ax.legend()

# Annotate final cumulative AR
ar_text = f'Empirical Attack Rate: {emp_final_AR:.3f}  |  Simulated Attack Rate: {final_AR:.3f}'
fig.text(0.5, 0.04, ar_text, ha='center', fontsize=12)

# Add overall interpretive comment
comment_text = ('Note: Model underestimates Attack Rate and epidemic duration compared to data. '
                'Simulated convalescent counts poorly match empirical data (low R²).')
fig.text(0.5, 0.01, comment_text, ha='center', fontsize=12, color='red')

# Save plot
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('output/sim_vs_empirical_overlay.png')
plt.close()