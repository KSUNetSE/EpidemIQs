
import matplotlib.pyplot as plt
import pandas as pd
import os
output_dir = os.path.join(os.getcwd(), 'output')
# plot aggregated
agg = pd.read_csv(os.path.join(output_dir, 'results-1-1.csv'))
low = pd.read_csv(os.path.join(output_dir, 'results-1-2.csv'))
plt.figure(figsize=(10,4))
plt.plot(agg['time'], agg['I'], label='Aggregated High-degree')
plt.plot(low['time'], low['I'], label='Instantaneous Low-degree')
plt.xlabel('Time')
plt.ylabel('Infected')
plt.legend()
plt.title('SIR model: Aggregated (static) vs. Instantaneous (static) networks')
plt.tight_layout()
fig_path = os.path.join(output_dir, 'sir-results-static.png')
plt.savefig(fig_path)
plt.close()
fig_path