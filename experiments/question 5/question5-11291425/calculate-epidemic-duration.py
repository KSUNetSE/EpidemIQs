
# Define threshold to consider epidemic ended: I/N < 0.001 (less than 10 infected)
threshold = 0.001

# Find last time when I/N is above threshold
epidemic_end_index = np.max(np.where(I_frac > threshold)[0])
epidemic_duration = data.loc[epidemic_end_index, 'time'] - data.loc[0, 'time']
epidemic_duration