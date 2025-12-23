
# Calculate total population N from initial state (time=0) to confirm
N = data.loc[0, ['S', 'I', 'R']].sum()

# Calculate final recovered fraction
final_R = data.loc[data.index[-1], 'R']
final_recovered_fraction = final_R / N

# Calculate peak infected fraction
peak_I = data['I'].max()
peak_infected_fraction = peak_I / N

N, final_recovered_fraction, peak_infected_fraction