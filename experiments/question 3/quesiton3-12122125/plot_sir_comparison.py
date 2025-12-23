
import matplotlib.pyplot as plt
import os

# Reduce temporal time series to first 101 points if needed (already 101 points)
temp_ts = time_series_temporal

# Sample static time series to reduce plotting overload (sample every 2000 points)
static_ts_sampled = static_time_series.iloc[::2000, :]

# Plot comparison of mean S, I, R and confidence intervals for temporal and static networks
plt.figure(figsize=(15, 10))

# Plot S
plt.subplot(3,1,1)
plt.fill_between(temp_ts['time'], temp_ts['S_5p'], temp_ts['S_95p'], color='tab:blue', alpha=0.3, label='Temporal 90% CI')
plt.plot(temp_ts['time'], temp_ts['S_mean'], 'b-', label='Temporal Mean')
plt.fill_between(static_ts_sampled['time'], static_ts_sampled['S_90ci_lower'], static_ts_sampled['S_90ci_upper'], color='tab:orange', alpha=0.3, label='Static 90% CI')
plt.plot(static_ts_sampled['time'], static_ts_sampled['S_mean'], 'r-', label='Static Mean')
plt.title('Susceptible (S) over time')
plt.ylabel('Count')
plt.legend()

# Plot I
plt.subplot(3,1,2)
plt.fill_between(temp_ts['time'], temp_ts['I_5p'], temp_ts['I_95p'], color='tab:blue', alpha=0.3, label='Temporal 90% CI')
plt.plot(temp_ts['time'], temp_ts['I_mean'], 'b-', label='Temporal Mean')
plt.fill_between(static_ts_sampled['time'], static_ts_sampled['I_90ci_lower'], static_ts_sampled['I_90ci_upper'], color='tab:orange', alpha=0.3, label='Static 90% CI')
plt.plot(static_ts_sampled['time'], static_ts_sampled['I_mean'], 'r-', label='Static Mean')
plt.title('Infectious (I) over time')
plt.ylabel('Count')
plt.legend()

# Plot R
plt.subplot(3,1,3)
plt.fill_between(temp_ts['time'], temp_ts['R_5p'], temp_ts['R_95p'], color='tab:blue', alpha=0.3, label='Temporal 90% CI')
plt.plot(temp_ts['time'], temp_ts['R_mean'], 'b-', label='Temporal Mean')
plt.fill_between(static_ts_sampled['time'], static_ts_sampled['R_90ci_lower'], static_ts_sampled['R_90ci_upper'], color='tab:orange', alpha=0.3, label='Static 90% CI')
plt.plot(static_ts_sampled['time'], static_ts_sampled['R_mean'], 'r-', label='Static Mean')
plt.title('Recovered (R) over time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()

plt.tight_layout()

# Save plot
output_dir = os.path.join(os.getcwd(), "output")
output_path = os.path.join(output_dir, 'sir_model_comparison.png')
plt.savefig(output_path)
plt.close()