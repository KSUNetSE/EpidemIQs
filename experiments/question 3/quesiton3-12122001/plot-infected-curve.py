
import matplotlib.pyplot as plt
import os

# Prepare plot data
I_lower = data['I_90ci_lower'].values
I_upper = data['I_90ci_upper'].values

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(time, I, label='Mean Infected (I)')
plt.fill_between(time, I_lower, I_upper, color='orange', alpha=0.3, label='90% CI')

# Annotate peak infection
plt.scatter([peak_I_time], [peak_I], color='red')
plt.annotate(f'Peak I: {peak_I:.2f} ({peak_I_percent:.2f}%)\nTime: {peak_I_time}',
             (peak_I_time, peak_I),
             textcoords="offset points", xytext=(-50,10), ha='center')

plt.title('Infected I(t) over time with 90% CI')
plt.xlabel('Time (units)')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.grid(True)

# Save plot
output_dir = os.path.join(os.getcwd(), 'output')
plot_path = os.path.join(output_dir, 'infected_curve_with_CI.png')
plt.savefig(plot_path)
plt.close()

plot_path