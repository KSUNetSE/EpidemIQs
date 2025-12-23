
import matplotlib.pyplot as plt
import os

# Create output folder if not exists
output_dir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot S, I, R with confidence intervals
plt.figure(figsize=(10,6))

plt.plot(time, S, label='Susceptible (S)', color='blue')
plt.fill_between(time, data['SIR_S_90ci_lower'], data['SIR_S_90ci_upper'], color='blue', alpha=0.2)

plt.plot(time, I, label='Infected (I)', color='red')
plt.fill_between(time, data['SIR_I_90ci_lower'], data['SIR_I_90ci_upper'], color='red', alpha=0.2)

plt.plot(time, R, label='Recovered (R)', color='green')
plt.fill_between(time, data['SIR_R_90ci_lower'], data['SIR_R_90ci_upper'], color='green', alpha=0.2)

plt.xlabel('Time (units)')
plt.ylabel('Number of Individuals')
plt.title('SIR Dynamics Over Time with 90% Confidence Intervals')
plt.legend()
plt.grid(True)

# Save plot
plot_path = os.path.join(output_dir, 'sir_dynamics_summary.png')
plt.savefig(plot_path)
plt.close()

plot_path