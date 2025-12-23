
import pandas as pd
import numpy as np

data = pd.read_csv('output/results-11.csv')
data2 = pd.read_csv('output/results-12.csv')

# Analytical metrics: Peak infection, total epidemic size, duration
peak_I = np.max(data['I'])
peak_time = data['t'][np.argmax(data['I'])]
final_size = data['R'].iloc[-1]
duration = data['t'][np.where(data['I'] < 1)[0][0]] if np.any(data['I'] < 1) else data['t'].iloc[-1]

# For scenario 2 (S=0):
t2 = data2['t']
I2 = data2['I_noS']
peak_I2 = I2.iloc[0]
peak_time2 = t2.iloc[0]
final_size2 = 10000 - I2.iloc[-1]
duration2 = t2[np.where(I2 < 1)[0][0]] if np.any(I2 < 1) else t2.iloc[-1]

results_dict = {
    'Metric': ['Peak I', 'Peak time', 'Final epidemic size', 'Duration'],
    'Decline in infectives': [peak_I, peak_time, final_size, duration],
    'No susceptibles (S=0)': [peak_I2, peak_time2, final_size2, duration2]
}
metric_table = pd.DataFrame(results_dict)
metric_table.to_csv('output/results-13.csv', index=False)
print(metric_table)
