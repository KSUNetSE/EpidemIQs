
import pandas as pd
import os

def describe_seir_curves():
    df_er = pd.read_csv(os.path.join(os.getcwd(),'output','results-1-1.csv'))
    df_ba = pd.read_csv(os.path.join(os.getcwd(),'output','results-1-2.csv'))
    # Identify key shape features
    summary = []
    # Peak infectious
    peak_er = df_er['I'].max(); t_peak_er = df_er['time'][df_er['I'].idxmax()]
    peak_ba = df_ba['I'].max(); t_peak_ba = df_ba['time'][df_ba['I'].idxmax()]
    summary.append(f"ER: Peak I={peak_er} at t={t_peak_er:.1f}. BA: Peak I={peak_ba} at t={t_peak_ba:.1f}.")
    # Duration (I>1)
    dur_er = df_er['time'][df_er['I']>1].max() - df_er['time'][df_er['I']>1].min()
    dur_ba = df_ba['time'][df_ba['I']>1].max() - df_ba['time'][df_ba['I']>1].min()
    summary.append(f"ER: Epidemic duration ~{dur_er:.1f}. BA: ~{dur_ba:.1f}.")
    # Early exponential phase: compare slope
    slope_er = (df_er['I'][5] - df_er['I'][1]) / (df_er['time'][5] - df_er['time'][1])
    slope_ba = (df_ba['I'][5] - df_ba['I'][1]) / (df_ba['time'][5] - df_ba['time'][1])
    summary.append(f"Early slope BA: {slope_ba:.1f} per unit time; ER: {slope_er:.1f}.")
    # Final size (total R)
    final_er = df_er['R'].iloc[-1]
    final_ba = df_ba['R'].iloc[-1]
    summary.append(f"ER final size (R): {final_er}, BA: {final_ba}.")
    summary.append("BA network shows a higher, earlier peak and faster initial take-off, but shorter overall epidemic duration and smaller final size.")
    return '\n'.join(summary)

describe_seir_curves()