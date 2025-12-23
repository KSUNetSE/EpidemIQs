
import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))

plt.figure(figsize=(8,6))
plt.plot(df['time'], df['S'], label='Susceptible', color='blue')
plt.plot(df['time'], df['I'], label='Infected', color='red')
plt.plot(df['time'], df['R'], label='Recovered', color='green')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Epidemic Dynamics on Static Small-World Network')
plt.legend()
plt.grid(True)
plt.xlim(df['time'].iloc[0], df['time'].iloc[-1])
plt.tight_layout()
describe = {}
describe['S0'] = df['S'][0]
describe['I0'] = df['I'][0]
describe['R0'] = df['R'][0]
describe['S_final'] = df['S'].iloc[-1]
describe['R_final'] = df['R'].iloc[-1]
describe['one_peak'] = int((df['I'].max() == df['I'].max()) and (df['I'].value_counts()[df['I'].max()] == 1))
return_vars = ['describe']