
# Find the time of near extinction (last time when I >= 10)
time_near_extinction = data.loc[data['I'] >= 10, 'time'].max()

# Duration of epidemic = time_near_extinction - starting time (0 assumed)
duration_of_epidemic = time_near_extinction - 0

time_near_extinction, duration_of_epidemic