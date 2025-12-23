
import pandas as pd
import matplotlib.pyplot as plt

# Load both results
res1 = pd.read_csv('output/results-11.csv')
res2 = pd.read_csv('output/results-12.csv')

plt.figure(figsize=(10,6))
plt.plot(res1['time'], res1['I'], label='Random Seed (3 infected)')
plt.plot(res2['time'], res2['I'], label='Hub Seed (3 infected)')
plt.xlabel('Time (days)')
plt.ylabel('Number of Infected')
plt.title('Comparison of Infection Evolution: Random vs Hub Seeding')
plt.legend()
plt.tight_layout()
fig_path = 'output/compare_infected.png'
plt.savefig(fig_path)
plt.close()
fig_path