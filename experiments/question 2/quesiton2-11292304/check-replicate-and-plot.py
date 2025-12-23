
import matplotlib.pyplot as plt

# Check if data contains run/replicate identifier columns
replicate_cols = [col for col in data.columns if 'run' in col or 'replicate' in col or 'sim' in col]

# Plot mean infected (I) over time
plt.figure(figsize=(10,6))
plt.plot(data['time'], data['I'], label='Mean Infected')
plt.fill_between(data['time'], data['I_90ci_lower'], data['I_90ci_upper'], color='b', alpha=0.2, label='90% CI')
plt.xlabel('Time (days)')
plt.ylabel('Number of Infected individuals')
plt.title('Mean Infection Curve with 90% CI')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/mean_infection_curve.png')
plt.close()

replicate_cols