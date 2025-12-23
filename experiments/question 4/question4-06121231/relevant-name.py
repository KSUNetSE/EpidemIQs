
data_samples[0].columns.tolist()
data_samples[1].columns.tolist()
data_samples[2].columns.tolist()
import pandas as pd

file_paths = ['output/results-00.csv', 'output/results-01.csv', 'output/results-02.csv']
data_samples = [pd.read_csv(fp) for fp in file_paths]
data_head_samples = [df.head() for df in data_samples]
data_columns_samples = [df.columns.tolist() for df in data_samples]

import numpy as np

def analyze_scenario(df):
    # Total time and last 10% time index
    total_points = len(df)
    last_10_percent_idx = int(total_points*0.9)

    # Steady-state mean over last 10%
    steady_state_means = df.iloc[last_10_percent_idx:].mean()

    # Max I1 and I2 and time they occur
    max_I1 = df['I1'].max()
    max_I2 = df['I2'].max()
    max_I1_time = df.loc[df['I1'].idxmax(), 'time']
    max_I2_time = df.loc[df['I2'].idxmax(), 'time']

    # Function to find time to steady state within 1% variation for 10 consecutive points
    def time_to_steady_state(series):
        for start_idx in range(total_points - 10):
            window = series[start_idx:start_idx+10]
            if np.max(window) - np.min(window) <= 0.01 * np.mean(window):
                return df.loc[start_idx, 'time']
        return None

    time_steady_I1 = time_to_steady_state(df['I1'])
    time_steady_I2 = time_to_steady_state(df['I2'])

    # Extinction of I1 and I2 (0 for last 10%)
    ext_I1 = (df['I1'].iloc[last_10_percent_idx:] == 0).all()
    ext_I2 = (df['I2'].iloc[last_10_percent_idx:] == 0).all()

    # Extinction time (first 0 after which stays 0)
    def extinction_time(series):
        zero_indices = series[series == 0].index
        for idx in zero_indices:
            if (series.loc[idx:] == 0).all():
                return df.loc[idx, 'time']
        return None

    ext_time_I1 = extinction_time(df['I1']) if ext_I1 else None
    ext_time_I2 = extinction_time(df['I2']) if ext_I2 else None

    # Coexistence at steady state
    coexist = (steady_state_means['I1'] > 0) and (steady_state_means['I2'] > 0)

    # Epidemic duration: time from initial infection (time I1 or I2 first > 0) to steady state or extinction time
    def initial_infection_time(series):
        infected = series[series > 0]
        return df.loc[infected.index[0], 'time'] if not infected.empty else None

    init_inf_I1 = initial_infection_time(df['I1'])
    init_inf_I2 = initial_infection_time(df['I2'])

    def epidemic_duration(init_inf, steady, ext_time):
        if ext_time is not None:
            return ext_time - init_inf
        elif steady is not None:
            return steady - init_inf
        else:
            return None

    epid_dur_I1 = epidemic_duration(init_inf_I1, time_steady_I1, ext_time_I1)
    epid_dur_I2 = epidemic_duration(init_inf_I2, time_steady_I2, ext_time_I2)

    return {
        'steady_state_means': steady_state_means[['S', 'I1', 'I2']].to_dict(),
        'max_I1': max_I1,
        'max_I2': max_I2,
        'max_I1_time': max_I1_time,
        'max_I2_time': max_I2_time,
        'time_steady_I1': time_steady_I1,
        'time_steady_I2': time_steady_I2,
        'ext_I1': ext_I1,
        'ext_I2': ext_I2,
        'ext_time_I1': ext_time_I1,
        'ext_time_I2': ext_time_I2,
        'coexist': coexist,
        'epid_dur_I1': epid_dur_I1,
        'epid_dur_I2': epid_dur_I2
    }

# Apply to dataset 0
metrics_00 = analyze_scenario(data_samples[0])
metrics_01 = analyze_scenario(data_samples[1])
metrics_02 = analyze_scenario(data_samples[2])
def format_metrics(metrics, scen_label):
    return {
        'Scenario': scen_label,
        'Steady-state Prevalence': {
            'S (population)': metrics['steady_state_means']['S'],
            'I1 (population)': metrics['steady_state_means']['I1'],
            'I2 (population)': metrics['steady_state_means']['I2'],
        },
        'Max Prevalence': {
            'Max I1 (population)': metrics['max_I1'],
            'Max I2 (population)': metrics['max_I2'],
            'Time Max I1 (time units)': metrics['max_I1_time'],
            'Time Max I2 (time units)': metrics['max_I2_time'],
        },
        'Time to Steady State': {
            'I1 (time units)': metrics['time_steady_I1'],
            'I2 (time units)': metrics['time_steady_I2'],
        },
        'Extinction': {
            'I1 Extinct': metrics['ext_I1'],
            'I2 Extinct': metrics['ext_I2'],
            'I1 Extinction Time (time units)': metrics['ext_time_I1'],
            'I2 Extinction Time (time units)': metrics['ext_time_I2'],
        },
        'Coexistence at steady state': metrics['coexist'],
        'Epidemic Duration': {
            'I1 (time units)': metrics['epid_dur_I1'],
            'I2 (time units)': metrics['epid_dur_I2'],
        }
    }

summary_00 = format_metrics(metrics_00, 'Scenario 00')
summary_01 = format_metrics(metrics_01, 'Scenario 01')
summary_02 = format_metrics(metrics_02, 'Scenario 02')

summary_00, summary_01, summary_02