
import numpy as np

# Extract columns into variables
times = data['time'].values
S = data['SIR_S'].values
I = data['SIR_I'].values
R = data['SIR_R'].values

# Final epidemic size approx as final mean recovered
final_epidemic_size = R[-1]  # unit: individuals

# Peak prevalence
peak_prevalence = I.max()  # unit: individuals
peak_index = I.argmax()
# Time to peak
time_to_peak = times[peak_index]  # time units

final_epidemic_size, peak_prevalence, time_to_peak