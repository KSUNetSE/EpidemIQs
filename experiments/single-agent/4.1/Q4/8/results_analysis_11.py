
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load results
df = pd.read_csv('/Users/hosseinsamaei/phd/gemf_llm/output/results-11.csv')
# Analyze for metrics
# Peak infection value and time
t_peak = df['I'].idxmax()
peak_I = df['I'].max()
# Final epidemic size = R(t_final)
final_R = df['R'].iloc[-1]
# Epidemic duration = time between first above zero and last nonzero I
t_start = df['I'].gt(0).idxmax()
t_end = df['I'].ne(0).cumsum().idxmax() # last nonzero index
duration = t_end - t_start
# Infection rise/fall slopes (infections at 25%, 50%, 75% of peak)
def find_crossing(val):
    # Returns first idx where I >= val
    return df['I'].ge(val).idxmax()
I_25 = 0.25*peak_I
I_50 = 0.5*peak_I
I_75 = 0.75*peak_I
t_25 = find_crossing(I_25)
t_50 = find_crossing(I_50)
t_75 = find_crossing(I_75)
# Infection rise time (from 25% to 75%):
rise_time = t_75 - t_25
# Decay: time from 75% of peak to post-peak 25%
I_descend = df['I'].iloc[t_peak:].le(I_25)
if I_descend.any():
    t_75_down = t_peak + df['I'].iloc[t_peak:].le(I_75).idxmax()
    t_25_down = t_peak + I_descend.idxmax()
    decay_time = t_25_down - t_75_down
else:
    decay_time = np.nan
# Plot annotated infection curve
plt.figure(figsize=(8,5))
plt.plot(df['time'], df['I'], label='Infected')
plt.axvline(df['time'][t_peak], color='r', ls='--', label='Peak')
plt.axhline(I_25, color='g', ls=':', lw=1)
plt.axhline(I_50, color='g', ls=':', lw=1)
plt.axhline(I_75, color='g', ls=':', lw=1)
plt.xlabel('Time')
plt.ylabel('Infected Individuals')
plt.title('Annotated Epidemic Infection Curve')
plt.legend()
ann_path = '/Users/hosseinsamaei/phd/gemf_llm/output/annotated_infections_11.png'
plt.savefig(ann_path)
