
import pandas as pd
import matplotlib.pyplot as plt

# Load simulation results
results_path = '/Users/hosseinsamaei/phd/gemf_llm/output/results-11.csv'
data = pd.read_csv(results_path)

plt.figure(figsize=(8,5))
plt.plot(data['time'], data['S'], label='Susceptible')
plt.plot(data['time'], data['I1'], label='Infected I1')
plt.plot(data['time'], data['I2'], label='Infected I2')
plt.xlabel('Time')
plt.ylabel('Number of Nodes')
plt.title('Competitive SI1SI2S Simulation Results')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/Users/hosseinsamaei/phd/gemf_llm/output/plot_results_check.png')

# Find the periods where I1 and I2 are both elevated (above 10)
crossover_t = data[(data['I1'] > 10) & (data['I2'] > 10)]['time'].values

# Compare dominance timeframes
I1_during_I2 = data.loc[data['I2'] > 10, 'I1'].mean()
I2_during_I1 = data.loc[data['I1'] > 10, 'I2'].mean()
suppression_pattern = 'mixed'
if I1_during_I2 > 2 * I2_during_I1:
    suppression_pattern = 'I1 dominates strongly'
elif I2_during_I1 > 2 * I1_during_I2:
    suppression_pattern = 'I2 dominates strongly'

coexist_duration = len(crossover_t) * (data['time'][1]-data['time'][0])

description = f"Overlap duration where both I1 and I2 >10: {coexist_duration:.1f} units.\n"
description += f"Average I1 when I2>10: {I1_during_I2:.1f}; Average I2 when I1>10: {I2_during_I1:.1f}.\n"
description += f"Suppression pattern: {suppression_pattern}.\n"

# Analyze which compartment is present at end
desc_end = f"Final: I1={data['I1'].iloc[-1]}, I2={data['I2'].iloc[-1]}. "
if data['I1'].iloc[-1]==0 and data['I2'].iloc[-1]==0:
    desc_end += 'Both extinct.'
elif data['I1'].iloc[-1]>0 and data['I2'].iloc[-1]==0:
    desc_end += 'Only I1 survives.'
elif data['I1'].iloc[-1]==0 and data['I2'].iloc[-1]>0:
    desc_end += 'Only I2 survives.'
else:
    desc_end += 'Both coexist.'

full_desc = description + desc_end
full_desc