
# Calculate total population N assuming constant population over time
N = (data['S'] + data['I'] + data['R']).iloc[0]

# Calculate final recovered fraction R(t_final)/N
final_recovered_fraction = data['R'].iloc[-1] / N

# Calculate peak infected fraction max I(t)/N
peak_infected_fraction = data['I'].max() / N

final_recovered_fraction, peak_infected_fraction