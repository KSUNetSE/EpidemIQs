
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load the results
data = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))

plt.figure(figsize=(10,5))
plt.plot(data['time'], data['S'], label='Susceptible')
plt.plot(data['time'], data['I'], label='Infected')
plt.plot(data['time'], data['R'], label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('Epidemic Evolution (SIR over ER Network, N=1000, <k>=8, R0=2.5)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(os.getcwd(), 'output', 'results-11-annotated.png')
plt.savefig(plot_path)
plt.close()
'Annotated plot saved.', plot_path