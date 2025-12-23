
import matplotlib.pyplot as plt

# Visualize I in early phase to decide on doubling time calculation window
early_phase = data[data['time'] <= 1.0]

plt.figure(figsize=(10, 6))
plt.plot(early_phase['time'], early_phase['I'], label='I(t) Early Phase')
plt.xlabel('Time')
plt.ylabel('Number of Infected Individuals (I)')
plt.title('Early Phase Infection Dynamics')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot in the output directory
plt.savefig('output/early_phase_infections.png')

plt.close()

"Early phase plot saved to output/early_phase_infections.png"