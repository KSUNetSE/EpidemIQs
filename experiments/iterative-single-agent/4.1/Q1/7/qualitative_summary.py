
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('output/results-12.csv')

# Plot epidemic curves
plt.figure(figsize=(8,5))
plt.plot(data['time'], data['S'], label='Susceptible')
plt.plot(data['time'], data['I'], label='Infected')
plt.plot(data['time'], data['R'], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Population in Each Compartment Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('output/epidemic_summary_plot.png')

# Analyze qualitative curve
I_max = data['I'].max()
S_init = data['S'].iloc[0]
R_end = data['R'].iloc[-1]
I_curve_shape = 'sharp peak' if I_max > 0.05*S_init else 'low peak'
final_outbreak = 'contained' if R_end < 0.1*S_init else 'large outbreak'
curve_observation = f"The epidemic curve had a {I_curve_shape} with a final outbreak that was {final_outbreak}.\nPeak infected: {I_max}, Final recovered: {R_end}" 
curve_observation
