
import pandas as pd
import numpy as np
import os

# Helper for key metrics extraction from simulations
def extract_metrics(csv_path):
    data = pd.read_csv(csv_path)
    # Find peak infection and time to peak
    peak_I = data['I'].max()
    peak_time = float(data[data['I'] == peak_I]['time'].min())
    # Final size: total recovered at end
    final_R = data['R'].iloc[-1]
    # Was epidemic sustained? If I drops to 0 - 'died out' else 'sustained'
    died_out = data['I'].iloc[-1] == 0
    duration = data['time'].iloc[-1]
    return {
        'Peak Infected': int(peak_I),
        'Time to Peak': float(peak_time),
        'Final Recovered': int(final_R),
        'Epidemic Died Out': bool(died_out),
        'Duration': float(duration)
    }

results10 = extract_metrics(os.path.join(os.getcwd(), 'output', 'results-10.csv'))
results11 = extract_metrics(os.path.join(os.getcwd(), 'output', 'results-11.csv'))
results12 = extract_metrics(os.path.join(os.getcwd(), 'output', 'results-12.csv'))

metrics_table = pd.DataFrame([
    {'Scenario': 'No Vaccination', **results10},
    {'Scenario': 'Random Vaccination', **results11},
    {'Scenario': 'Degree-10 Vaccination', **results12},
])
metrics_table.to_csv(os.path.join(os.getcwd(), 'output', 'summary_metrics.csv'), index=False)
metrics_table