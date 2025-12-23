
import pandas as pd
import matplotlib.pyplot as plt

# Data from above
labels = ['Aggregated Static', 'Activity-Driven Temporal']
final_size = [0.9405, 0.6583] # As computed above
x = range(len(labels))

plt.figure(figsize=(6,4))
plt.bar(x, final_size, color=['#3366cc','#ff9900'])
plt.ylabel('Final Epidemic Size (Fraction Infected)')
plt.xticks(x, labels)
plt.ylim(0,1)
for i, v in enumerate(final_size):
    plt.text(i, v + 0.025, f'{v:.2f}', ha='center', fontweight='bold')
plt.title('Final Epidemic Size: Temporal vs Aggregated Network (SIR, $R_0=3$)')
plt.tight_layout()
plt.savefig('output/results-11.png')

# Table for summary
summary = pd.DataFrame({
    'Network Type': labels,
    'Final Epidemic Size': final_size
})
summary.to_csv('output/results-11.csv', index=False)