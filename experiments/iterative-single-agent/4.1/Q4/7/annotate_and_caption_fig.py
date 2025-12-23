
from matplotlib import pyplot as plt
import pandas as pd
import os

data = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))
plt.figure(figsize=(8,4))
plt.plot(data['time'], data['S'], label='Susceptible')
plt.plot(data['time'], data['I'], label='Infected')
plt.plot(data['time'], data['R'], label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('Population evolution in SIR Model over ER Network')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-11-annotated.png'))
plt.close()
# Provide caption
caption = 'Figure shows population evolution in each compartment (Susceptible, Infected, Recovered) for SIR model simulation over Erdős–Rényi network (N=1000, <k>~10). Peak infection at ~20% occurs around day 27. Epidemic ends in 73 days with 79% ultimately infected. Susceptible population sharply decreases after day 20, while recovery dominates after day 40.'
caption