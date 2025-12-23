
import matplotlib.pyplot as plt
import pandas as pd
import os
# Quick visualize all three epidemic trajectories
files = ['results-10.csv', 'results-11.csv', 'results-12.csv']
labels = ['No Vaccination', 'Random Vaccination', 'Degree-10 Vaccination']
plt.figure(figsize=(10,6))
for f,label in zip(files, labels):
    df = pd.read_csv(os.path.join(os.getcwd(), 'output', f))
    plt.plot(df['time'], df['I'], label=label)
plt.xlabel('Time')
plt.ylabel('# Infected')
plt.title('Comparison of Infection Curves Across Vaccination Strategies')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'output','fig-comparison-infection.png'))
plt.close()