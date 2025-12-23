
import numpy as np

N = 10000

# Final epidemic size (last row of R)
final_R_abs = data['R'].iloc[-1]
final_R_frac = final_R_abs / N

# Maximum infectious and timing
max_I = data['I'].max()
peak_day = data.loc[data['I'].idxmax(), 'time']

# Epidemic duration
nonzero_I = data[data['I'] >= 1]
epidemic_start = nonzero_I['time'].iloc[0]
epidemic_end = nonzero_I['time'].iloc[-1]
epidemic_duration = epidemic_end - epidemic_start

# Number of E, I at t=0
initial_E = data['E'].iloc[0]
initial_I = data['I'].iloc[0]

final_R_abs, final_R_frac, max_I, peak_day, epidemic_start, epidemic_end, epidemic_duration, initial_E, initial_I