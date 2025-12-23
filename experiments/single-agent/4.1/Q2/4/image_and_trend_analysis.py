
import matplotlib.pyplot as plt
import pandas as pd

# Load simulation data
results = pd.read_csv('output/results-11.csv')
plt.figure(figsize=(8,5))
plt.plot(results['time'], results['S'], label='Susceptible')
plt.plot(results['time'], results['I'], label='Infected')
plt.plot(results['time'], results['R'], label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Dynamics on Scale-Free Network (Barabasi-Albert)')
plt.legend()
plt.tight_layout()
plt.savefig('output/results-11-review.png')

# Analysis for qualitative trends
single_peak = (results['I'].max() == max(results['I'])) and (sum((results['I'].diff().dropna() > 0).astype(int)) == 1)
recovery_complete = results['I'].iloc[-1] == 0

description = ''
if single_peak:
    description += 'The epidemic exhibits a single infection wave with a clear peak, after which infections decline rapidly. '
else:
    description += 'Multiple peaks or a prolonged tail in infection numbers is present. '
if recovery_complete:
    description += 'By the simulation end, recovery is nearly complete with almost no infected individuals remaining.'
else:
    description += 'The epidemic remains active at the end of the simulation, with some individuals still infected.'

description
