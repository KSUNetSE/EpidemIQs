
# Check min and max values of S, I, R over entire time
min_S = data['SIR_S'].min()
max_S = data['SIR_S'].max()
min_I = data['SIR_I'].min()
max_I = data['SIR_I'].max()
min_R = data['SIR_R'].min()
max_R = data['SIR_R'].max()

# Check sums at initial and final time points
initial_sum = data.loc[0, ['SIR_S', 'SIR_I', 'SIR_R']].sum()
final_sum = data.loc[data.index[-1], ['SIR_S', 'SIR_I', 'SIR_R']].sum()

min_S, max_S, min_I, max_I, min_R, max_R, initial_sum, final_sum