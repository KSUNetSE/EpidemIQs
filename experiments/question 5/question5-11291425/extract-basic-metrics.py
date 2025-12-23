
import numpy as np

# Parameters
total_population = 10000

# Normalize R and I by population size
R_frac = data['R'] / total_population
I_frac = data['I'] / total_population

# Calculate final size: max R/N
final_size = R_frac.max()

# Calculate peak infection prevalence: max I/N
peak_infection = I_frac.max()

# Time of peak infection
time_of_peak = data.loc[I_frac.idxmax(), 'time']

final_size, peak_infection, time_of_peak