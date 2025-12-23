
import numpy as np

N = 1000

# Final recovered fraction
final_recovered = data['R'].iloc[-1]
final_recovered_fraction = final_recovered / N

# Peak number of infected and time to peak
peak_infected = data['I'].max()
time_to_peak = data.loc[data['I'].idxmax(), 'time']

# Epidemic duration: time when I first reaches zero and stays zero afterwards
infected_zero_indices = np.where(data['I'] == 0)[0]
epidemic_duration = None
for idx in infected_zero_indices:
    if (data['I'].iloc[idx:] == 0).all():
        epidemic_duration = data['time'].iloc[idx]
        break

final_recovered_fraction, peak_infected, time_to_peak, epidemic_duration