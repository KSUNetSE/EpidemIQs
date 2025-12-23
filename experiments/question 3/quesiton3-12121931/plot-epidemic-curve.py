
import matplotlib.pyplot as plt

# Plot to visually summarize the epidemic curve (I over time) for the given data
plt.figure(figsize=(8,5))
plt.plot(data['time'], data['I'], label='Infected (I)')
plt.fill_between(data['time'], data['I_90ci_lower'], data['I_90ci_upper'], alpha=0.2, label='I 90% CI')
plt.xlabel('Time')
plt.ylabel('Number of Infected Individuals')
plt.title('Epidemic Curve (Temporal SIR Simulation)')
plt.legend()
plt.tight_layout()

# Save the plot to output directory
plt.savefig('output/temporal_SIR_epidemic_curve.png')

# Simple textual summary of shape
epidemic_curve_shape_summary = "Rapid die-out with no significant epidemic spread, I quickly falls to near zero after start."

epidemic_curve_shape_summary