
N = 10000

# Calculate final epidemic size as max R/N
final_epidemic_size = data['R'].max() / N

# Calculate peak infection prevalence as max I/N
peak_infection_prevalence = data['I'].max() / N

# Find time of peak infection
max_I_row = data.loc[data['I'].idxmax()]
time_of_peak_infection = max_I_row['time']

final_epidemic_size, peak_infection_prevalence, time_of_peak_infection