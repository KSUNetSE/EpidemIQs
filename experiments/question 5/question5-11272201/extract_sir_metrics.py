
import numpy as np
import pandas as pd
import os

filepaths = [os.path.join('output', f'results-{i}0.csv') for i in range(5)]
N = 10000

results = {}

for fp in filepaths:
    df = pd.read_csv(fp)
    
    # Calculate fraction vaccinated from initial S
    S0 = df.loc[0, 'S']
    frac_vaccinated = 1 - S0 / N
    
    # Final epidemic size: final R
    R_final = df.loc[df.index[-1], 'R']
    R_final_lower = df.loc[df.index[-1], 'R_90ci_lower']
    R_final_upper = df.loc[df.index[-1], 'R_90ci_upper']
    R_final_std = (R_final_upper - R_final_lower) / (2*1.645)
    R_final_frac = R_final / N
    R_final_frac_std = R_final_std / N
    R_final_95ci = (R_final - 1.96 * R_final_std, R_final + 1.96 * R_final_std)
    R_final_frac_95ci = (R_final_frac - 1.96 * R_final_frac_std, R_final_frac + 1.96 * R_final_frac_std)
    
    # Peak infection prevalence and time
    I_max = df['I'].max()
    I_max_row = df['I'].idxmax()
    I_max_time = df.loc[I_max_row, 'time']
    I_max_lower = df.loc[I_max_row, 'I_90ci_lower']
    I_max_upper = df.loc[I_max_row, 'I_90ci_upper']
    I_max_std = (I_max_upper - I_max_lower) / (2*1.645)
    I_max_frac = I_max / N
    I_max_frac_std = I_max_std / N
    I_max_95ci = (I_max - 1.96 * I_max_std, I_max + 1.96 * I_max_std)
    I_max_frac_95ci = (I_max_frac - 1.96 * I_max_frac_std, I_max_frac + 1.96 * I_max_frac_std)
    
    # Epidemic duration: last time where I > 0
    non_zero_I = df[df['I'] > 0]
    epidemic_duration = non_zero_I['time'].max() if not non_zero_I.empty else 0

    # Store results
    results[os.path.basename(fp)] = {
        'fraction_vaccinated': frac_vaccinated,
        'final_epidemic_size': {'mean': R_final, 'std': R_final_std, '95ci': R_final_95ci, 'fraction': R_final_frac, 'fraction_std': R_final_frac_std, 'fraction_95ci': R_final_frac_95ci},
        'peak_infection': {'mean': I_max, 'std': I_max_std, '95ci': I_max_95ci, 'fraction': I_max_frac, 'fraction_std': I_max_frac_std, 'fraction_95ci': I_max_frac_95ci},
        'peak_time': {'mean': I_max_time},
        'epidemic_duration': {'mean': epidemic_duration},
    }

results