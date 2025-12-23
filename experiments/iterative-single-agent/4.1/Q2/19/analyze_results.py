
# Load and analyze both SIR (results-1-1.csv) and SI (results-1-2.csv) simulation results
import pandas as pd
sir_csv = 'output/results-1-1.csv'
si_csv = 'output/results-1-2.csv'
sir_df = pd.read_csv(sir_csv)
si_df = pd.read_csv(si_csv)

# SIR metrics
N = 500
metrics = {}

# Final size (epidemic final attack rate): total fraction infected = N-R(final) - S(final)
sir_final_S = sir_df['S'].iloc[-1]
sir_final_R = sir_df['R'].iloc[-1]
sir_final_I = sir_df['I'].iloc[-1]
final_attack_rate = (sir_final_R) / N
# Peak infection (max I)
peak_I = sir_df['I'].max()/N
peak_time = sir_df.loc[sir_df['I'].idxmax(),'time']

# Duration: time until I falls below 1/N threshold (here = 1)
try:
    end_time = sir_df[sir_df['I'] < 1]['time'].iloc[0]
except IndexError:
    end_time = sir_df['time'].iloc[-1]

metrics['sir'] = {'final_attack_rate':final_attack_rate,'peak_I':peak_I,'peak_time':peak_time,'end_time':end_time,
                  'susceptibles_remaining':sir_final_S/N,'infectives_remaining':sir_final_I/N}

# SI metrics
si_final_I = si_df['I'].iloc[-1]
si_final_S = si_df['S'].iloc[-1]
metrics['si'] = {'final_attack_rate':si_final_I/N,'susceptibles_remaining':si_final_S/N}

metrics