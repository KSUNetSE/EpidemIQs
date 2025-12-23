
# Confirming unique time points and time range
unique_times = data['time'].nunique()
time_min = data['time'].min()
time_max = data['time'].max()

# Basic overview of I to guess peak and final values
I_peak = data['I'].max()
t_peak = data.loc[data['I'].idxmax(), 'time']
I_final = data['I'].iloc[-1]

# Final values of S and R (last time point)
S_final = data['S'].iloc[-1]
R_final = data['R'].iloc[-1]

# Duration in time units
duration = time_max - time_min

unique_times, time_min, time_max, I_peak, t_peak, I_final, S_final, R_final, duration