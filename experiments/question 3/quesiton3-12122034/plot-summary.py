
# Manually define the extracted metrics from previous steps
max_I_mean = 3.602
time_to_peak_mean = 0.460
final_R_mean = 5.568
duration_mean = 1.971
fade_out_possible = True

doubling_time_val = 0.093

import matplotlib.pyplot as plt
import os

plt.figure(figsize=(10,6))

# Plot S, I, R with confidence intervals
plt.fill_between(data['time'], data['SIR_S_90ci_lower'], data['SIR_S_90ci_upper'], color='blue', alpha=0.1)
plt.fill_between(data['time'], data['SIR_I_90ci_lower'], data['SIR_I_90ci_upper'], color='red', alpha=0.1)
plt.fill_between(data['time'], data['SIR_R_90ci_lower'], data['SIR_R_90ci_upper'], color='green', alpha=0.1)

plt.plot(data['time'], data['SIR_S'], label='Susceptible', color='blue')
plt.plot(data['time'], data['SIR_I'], label='Infected', color='red')
plt.plot(data['time'], data['SIR_R'], label='Recovered', color='green')

# Mark peak infected
plt.scatter([time_to_peak_mean], [max_I_mean], color='black', s=50, label='Peak Infected')

# Annotations for summary statistics
plt.text(0.5, 900, f"Final Epidemic Size Mean: {final_R_mean:.2f}", fontsize=12, color='green')
plt.text(0.5, 850, f"Max Infected Mean: {max_I_mean:.2f}", fontsize=12, color='red')
plt.text(0.5, 800, f"Time to Peak Mean: {time_to_peak_mean:.2f}", fontsize=12)
plt.text(0.5, 750, f"Duration Mean: {duration_mean:.2f}", fontsize=12)
plt.text(0.5, 700, f"Early Fade-out: {'Yes' if fade_out_possible else 'No'}", fontsize=12)
plt.text(0.5, 650, f"Doubling Time (early): {doubling_time_val:.3f}", fontsize=12)

plt.xlabel('Time (units)')
plt.ylabel('Number of individuals')
plt.title('SIR Epidemic Dynamics with Confidence Intervals')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
output_path = os.path.join(os.getcwd(), 'output', 'SIR_epidemic_summary.png')
plt.savefig(output_path)
plt.close()

output_path