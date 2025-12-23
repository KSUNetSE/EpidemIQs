
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import os

# Filter for early 10 days and I > 1 to avoid log(0)
early_data = data[(data['time'] >= 0) & (data['time'] <= 10) & (data['I'] > 1)][['time', 'I']]

# Calculate log(I)
early_data['logI'] = np.log(early_data['I'])

# Linear regression on log(I) vs time
growth_model = linregress(early_data['time'], early_data['logI'])

# Growth rate (slope)
growth_rate = growth_model.slope

# Empirical doubling time = ln(2)/growth rate
if growth_rate > 0:
    doubling_time = np.log(2) / growth_rate
else:
    doubling_time = np.inf

# Plot I(t) with peak time and peak I marked
plot_data = data[['time', 'I']].copy()

plt.figure(figsize=(10,6))
plt.plot(plot_data['time'], plot_data['I'], label='Infectious I(t)')
plt.scatter([peak_time], [peak_I], color='red', label=f'Peak I: {peak_I:.1f} at {peak_time:.1f} days')
plt.axvline(x=peak_time, color='red', linestyle='--')
plt.title('Infectious Population Over Time')
plt.xlabel('Time (days)')
plt.ylabel('Number of Infectious Individuals')
plt.legend()
plt.grid(True)

# Save plot
o_dir = os.path.join(os.getcwd(), 'output')
plot_file = os.path.join(o_dir, 'Infectious_I_time_series.png')
plt.savefig(plot_file)

plot_file, growth_rate, doubling_time