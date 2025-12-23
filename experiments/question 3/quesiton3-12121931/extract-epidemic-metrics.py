
import numpy as np

# Total and fraction of R at final time
final_time = data['time'].iloc[-1]
final_R = data['R'].iloc[-1]
final_R_fraction = final_R / 1000  # since total nodes = 1000

# Peak infection rate and corresponding time
peak_I = data['I'].max()
peak_time = data['time'][data['I'].idxmax()]

# Epidemic duration: time until I falls below 1
# We consider duration as time at last index where I >=1
infected_above_one = data[data['I'] >= 1]
epidemic_duration = infected_above_one['time'].iloc[-1]

# Outbreak probability: fraction of runs with final R > 10
# The dataset appears aggregated over runs, so hypothetical runs data isn't directly available.
# Instead, if the confidence interval lower bound for R at final time is > 10, it suggests almost all runs had outbreak.
outbreak_prob_estimation = (data['R_90ci_lower'].iloc[-1] > 10)

# Additional observations
max_R = data['R'].max()
min_I = data['I'].min()

final_time, final_R, final_R_fraction, peak_I, peak_time, epidemic_duration, outbreak_prob_estimation, max_R, min_I