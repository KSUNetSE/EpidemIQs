
import numpy as np
from scipy.stats import linregress

N = 5000

# Helper function to find epidemic duration
# Defined as the time from first nonzero I to last time I < 1

def epidemic_duration(time, I):
    first_nonzero_idx = np.argmax(I > 0)
    # find last time I < 1 after peak
    last_below_1_idx = np.max(np.where(I < 1)[0])
    duration = time[last_below_1_idx] - time[first_nonzero_idx]
    return duration

# Helper function to estimate early exponential growth rate from log(I) vs time
# Using linear regression on initial phase where I > 0 and below peak

def early_growth_rate(time, I):
    # Find indices where I > 0
    pos_idx = np.where(I > 0)[0]
    if len(pos_idx) < 2:
        return np.nan  # Not enough points for regression

    # Limit to the early phase before peak (max I)
    peak_idx = np.argmax(I)
    early_idx = pos_idx[pos_idx < peak_idx]

    if len(early_idx) < 2:
        return np.nan

    # Log transform I
    log_I = np.log(I[early_idx])
    slope, intercept, r_value, p_value, std_err = linregress(time[early_idx], log_I)
    return slope, std_err


# Extract metrics for a given dataframe

def extract_metrics(df):
    time = df['time'].values
    I = df['I'].values
    R = df['R'].values
    I_90ci_lower = df['I_90ci_lower'].values
    I_90ci_upper = df['I_90ci_upper'].values
    R_90ci_lower = df['R_90ci_lower'].values
    R_90ci_upper = df['R_90ci_upper'].values

    # Epidemic duration
    duration = epidemic_duration(time, I)

    # Peak prevalence and time
    peak_idx = np.argmax(I)
    peak_I = I[peak_idx]
    peak_time = time[peak_idx]
    peak_I_90ci = (I_90ci_lower[peak_idx], I_90ci_upper[peak_idx])

    # Attack rate and CIs
    final_R = R[-1]
    attack_rate = final_R / N
    attack_rate_CI = (R_90ci_lower[-1]/N, R_90ci_upper[-1]/N)

    # Early exponential growth rate
    growth_rate, growth_rate_se = early_growth_rate(time, I)

    return {
        'Duration_days': duration,
        'Peak_prevalence': peak_I,
        'Peak_prevalence_90CI': peak_I_90ci,
        'Peak_time_days': peak_time,
        'Attack_rate': attack_rate,
        'Attack_rate_90CI': attack_rate_CI,
        'Early_growth_rate_per_day': growth_rate,
        'Early_growth_rate_SE': growth_rate_se
    }

metrics_er = extract_metrics(df_er)
metrics_ba = extract_metrics(df_ba)

metrics_er, metrics_ba