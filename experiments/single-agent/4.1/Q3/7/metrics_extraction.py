
import pandas as pd
import numpy as np
import os

def extract_metrics(filename):
    df = pd.read_csv(filename)
    # Find when I peaks (time, value)
    peak_idx = np.argmax(df['I'])
    time_peak = int(df['time'][peak_idx])
    I_peak = int(df['I'][peak_idx])
    # Epidemic duration: when I>0 (between first and last nonzero)
    nonzero = np.where(df['I'].to_numpy()>0)[0]
    t_start = int(df['time'][nonzero[0]])
    t_end = int(df['time'][nonzero[-1]])
    epic_dur = t_end-t_start
    # Final epidemic size: R at last time
    final_R = int(df['R'].iloc[-1])
    # Total infected = final R + whatever left in I at last step
    total_cases = int(df['R'].iloc[-1] + df['I'].iloc[-1])
    return {
        'peak_time': time_peak,
        'peak_infected': I_peak,
        'epidemic_duration': epic_dur,
        'final_epidemic_size_R': final_R,
        'total_cases': total_cases,
        't_start': t_start,
        't_end': t_end
    }

static_metrics = extract_metrics(os.path.join(os.getcwd(), 'output', 'results-10.csv'))
temp_metrics = extract_metrics(os.path.join(os.getcwd(), 'output', 'results-11.csv'))
all_metrics = {'static': static_metrics, 'temporal': temp_metrics}

import json
with open(os.path.join(os.getcwd(), 'output', 'metrics_summary.json'), 'w') as f:
    json.dump(all_metrics, f, indent=2)
all_metrics
