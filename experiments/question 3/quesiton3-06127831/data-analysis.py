
N = 1000

# Final recovered fraction R(t_final)/N
final_recovered_fraction = data['R'].iloc[-1] / N

# Peak number of infected max I(t)
peak_infected = data['I'].max()

# Time-to-peak t at which I(t) is maximal
time_to_peak = data['time'][data['I'].idxmax()]

# Epidemic duration: time when I(t) first reaches zero and remains zero
infected_zeros = data[data['I'] == 0]['time']
epidemic_duration = infected_zeros.iloc[0] if not infected_zeros.empty else None

final_recovered_fraction, peak_infected, time_to_peak, epidemic_duration