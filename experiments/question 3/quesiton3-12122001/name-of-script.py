
import pandas as pd

# Load the data from CSV to examine its structure
file_path = 'output/results-11.csv'
data = pd.read_csv(file_path)

# Inspect the first few rows and columns
head = data.head()
info = data.info(verbose=True)
shape = data.shape
columns = data.columns.to_list()
import numpy as np

# Constants
N = 1000

# Epidemic duration: find last time I > 0
infected_nonzero = data[data['I'] > 0]
epidemic_duration = infected_nonzero['time'].iloc[-1]

# Peak infection and time to peak
peak_row = data.loc[data['I'].idxmax()]
peak_infection = peak_row['I']
time_to_peak = peak_row['time']
peak_infection_pct = peak_infection / N * 100

# Final epidemic size and attack rate
final_r = data['R'].iloc[-1]
attack_rate = final_r / N * 100

# Doubling time from early exponential phase
# Select early phase where I is between initial I and about 10x initial
initial_infected = data['I'].iloc[0]
early_phase = data[(data['I'] > initial_infected) & (data['I'] < 10*initial_infected)]

# Use linear regression on log(I) vs time for early phase
data_for_reg = early_phase.loc[:, ['time', 'I']].copy()
data_for_reg = data_for_reg[data_for_reg['I'] > 0]
data_for_reg['log_I'] = np.log(data_for_reg['I'])

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(data_for_reg['time'], data_for_reg['log_I'])

# Doubling time formula: Td = ln(2) / growth rate (slope)
doubling_time = np.log(2) / slope if slope > 0 else np.inf

data_for_reg.head()
import matplotlib.pyplot as plt
import os

# Plot I(t) with key points
plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['I'], label='Mean Infected I(t)', color='blue')

# Mark key points
plt.scatter(time_to_peak, peak_infection, color='red', label=f'Peak Infection: {peak_infection:.2f} at day {time_to_peak}')
plt.axvline(epidemic_duration, color='green', linestyle='--', label=f'Epidemic duration: {epidemic_duration} days')

# Labels and legend
plt.title('Mean Infection Trajectory I(t) with Key Metrics')
plt.xlabel('Time (days)')
plt.ylabel('Number of Infected Individuals')
plt.legend()
plt.grid(True)

# Save plot
output_dir = os.path.join(os.getcwd(), 'output')
plot_path = os.path.join(output_dir, 'mean_infected_trajectory_with_metrics.png')
plt.savefig(plot_path)
plt.close()

plot_path