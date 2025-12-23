
import pandas as pd
import matplotlib.pyplot as plt
import os
# Overlay B(t) curves for all runs for visual comparison (simulated)
outputs = [
    ('results-11.csv', 'SBM baseline'),
    ('results-12.csv', 'SBM beta-10%'),
    ('results-13.csv', 'SBM beta+10%'),
    ('results-14.csv', 'SBM sigma=0.7 (+0.5d)'),
    ('results-15.csv', 'Seed 3 houses'),
    ('results-16.csv', 'ER baseline')
]
plt.figure(figsize=(10,6))
for fname, lbl in outputs:
    pth = os.path.join(os.getcwd(), 'output', fname)
    df = pd.read_csv(pth)
    if 'B' in df:
        plt.plot(df['time'], df['B'], label=lbl)
plt.xlabel('Time (days)')
plt.ylabel('Mean Confined to Bed (B)')
plt.title('Simulated Confined-to-Bed Prevalence Curves (Multiple Scenarios)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-sim-B-overlap.png'))
plt.close()
