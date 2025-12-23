
import pandas as pd
import numpy as np
import os
output_dir = os.path.join(os.getcwd(), 'output')

# Load result data
res_ER = pd.read_csv(os.path.join(output_dir, 'results-1-ER.csv'))
res_BA = pd.read_csv(os.path.join(output_dir, 'results-1-BA.csv'))

# Helper: extract metrics from results
def extract_epidemic_metrics(df, population=1000):
    peak_I = df['I'].max() / population
    time_to_peak = df['I'].idxmax()
    final_size = df['R'].iloc[-1] / population
    epidemic_duration = (df['I'] > 0).sum() # days with any infection
    doubling_time = np.nan
    max_infected = df['I'].max()
    # Estimate doubling time in early outbreak (first tripling of case)
    onset = df[df['I'] >= 1].index[0]
    for delta in range(2, 15):
        if onset + delta < len(df) and df['I'].iloc[onset + delta] >= 3 * df['I'].iloc[onset]:
            doubling_time = delta / np.log2(3)
            break
    return {
        'peak_infection_fraction': peak_I,
        'time_to_peak': float(df['time'].iloc[time_to_peak]),
        'final_epidemic_size': final_size,
        'epidemic_duration': epidemic_duration,
        'doubling_time': doubling_time,
        'max_infected': int(max_infected)
    }

metrics_ER = extract_epidemic_metrics(res_ER, population=1000)
metrics_BA = extract_epidemic_metrics(res_BA, population=1000)

all_metrics = {'ER': metrics_ER, 'BA': metrics_BA}

# Compose summary and rationale
summary = (
    "For SIR simulations with R0=2.5 and 1% initial infection: "
    f"\n- ER (random) network: peak infection fraction {metrics_ER['peak_infection_fraction']:.3f} at day {metrics_ER['time_to_peak']:.1f}, final size {metrics_ER['final_epidemic_size']:.3f}, epidemic duration {metrics_ER['epidemic_duration']}d.\n"
    f"- BA (scale-free) network: peak infection fraction {metrics_BA['peak_infection_fraction']:.3f} at day {metrics_BA['time_to_peak']:.1f}, final size {metrics_BA['final_epidemic_size']:.3f}, epidemic duration {metrics_BA['epidemic_duration']}d."
)

analysis_reasoning = ('Extracted classical metrics: peak infection, final epidemic size, time to peak, ' 
                      'duration, and early doubling time, reflecting severity and speed. Comparison ' 
                      'informs how topology alters outcomes despite identical mean connectivity and R0. ' 
                      'Metrics follow literature best practice for SIR on static networks.')
