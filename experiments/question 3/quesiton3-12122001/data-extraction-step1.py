
import numpy as np

# Extract relevant data columns
time = data['time'].values
I = data['I'].values
R = data['R'].values
N = 1000  # total population

# 1) Epidemic duration: time till I=0
# Since I may not hit exactly 0, take the last time where I>0
infected_nonzero_indices = np.where(I > 0)[0]
epidemic_duration = time[infected_nonzero_indices[-1]] if infected_nonzero_indices.size > 0 else 0

# 2) Peak infection (max I and % N)
peak_I = np.max(I)
peak_I_time = time[np.argmax(I)]
peak_I_percent = peak_I / N * 100

# 3) Time-to-peak
# This is peak_I_time from above

data
