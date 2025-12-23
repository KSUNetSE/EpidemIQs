
# Check initial sum of S, E, I, R to see if values are proportions or counts
total_initial = data.loc[0, ['S', 'E', 'I', 'R']].sum()

# Check max values of S, E, I, R for scale comparison
max_S = data['S'].max()
max_E = data['E'].max()
max_I = data['I'].max()
max_R = data['R'].max()

total_initial, max_S, max_E, max_I, max_R