
import pandas as pd
simdata = pd.read_csv('output/results-11.csv')
final = simdata.iloc[-1]
sumfinal = final['S'] + final['I'] + final['R']
metrics = {
    'S_final': int(final['S']),
    'I_final': int(final['I']),
    'R_final': int(final['R']),
    'Total_infected_fraction': float(final['R'])/sumfinal,
    'Epidemic_duration': simdata['time'].iloc[-1]
}
metrics