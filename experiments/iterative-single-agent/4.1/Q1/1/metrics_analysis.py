
import pandas as pd
import numpy as np
import os

def epidemic_metrics(csv_path):
    df = pd.read_csv(csv_path)
    # final size: total R at end minus initial R
    final_size = df['R'].iloc[-1] - df['R'].iloc[0]
    # peak infected
    peak_I = df['I'].max()
    time_to_peak = df['time'][df['I'].idxmax()]
    # epidemic duration until <1 infectious
    thresh = 1
    try:
        end_idx = np.where(df['I'] < thresh)[0][0]
        duration = df['time'][end_idx]
    except:
        duration = df['time'].iloc[-1]
    metrics = {
        'final_size': final_size,
        'peak_I': peak_I,
        'time_to_peak': time_to_peak,
        'duration': duration
    }
    return metrics

metrics_hom = epidemic_metrics('output/results-1-1.csv')
metrics_het = epidemic_metrics('output/results-1-2.csv')
(metrics_hom, metrics_het)