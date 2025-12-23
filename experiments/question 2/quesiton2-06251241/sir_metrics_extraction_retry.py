
# It seems there was an error because at some run I or other arrays might be empty or I[0] might not exist
# Need to add protection for empty or invalid data segments in the runs and also check run length

valid_runs = []
for run_df in runs:
    if len(run_df) > 0:
        valid_runs.append(run_df)

results_clean = {
    'epidemic_duration': [],
    'peak_prevalence': [],
    'peak_time': [],
    'final_epidemic_size': [],
    'doubling_time': [],
    'time_Re_below_1': [],
    'initial_S': [],
    'final_S': [],
    'outbreak': []
}

from scipy.stats import linregress
import numpy as np

for run_df in valid_runs:
    S = run_df['S'].values
    I = run_df['I'].values
    R = run_df['R'].values
    time = run_df['time'].values
    N = S[0] + I[0] + R[0]
    
    if len(I) == 0 or I[0] == 0:
        # No outbreak runs
        epidemic_duration = 0
        peak_prev = 0
        peak_time = np.nan
        final_size = R[-1] if len(R) > 0 else 0
        doubling_time = np.nan
        time_Re_below_1 = np.nan
        init_S = S[0] if len(S) > 0 else np.nan
        final_S = S[-1] if len(S) > 0 else np.nan
        outbreak = 0
    else:
        # Epidemic duration
        indices_nonzero_I = np.where(I > 0)[0]
        epidemic_duration = time[indices_nonzero_I[-1]]
        # Peak prevalence and time
        peak_idx = np.argmax(I)
        peak_prev = I[peak_idx]
        peak_time = time[peak_idx]
        # Final epidemic size
        final_size = R[-1]
        # Initial and final S
        init_S = S[0]
        final_S = S[-1]
        # Outbreak
        outbreak = 1
        
        # Doubling time estimation
        try:
            log_I = np.log(I[I>0])
            times_nonzero = time[I>0]
            lin_fit = linregress(times_nonzero, log_I)
            r = lin_fit.slope
            doubling_time = np.log(2) / r if r > 0 else np.nan
            # Estimate R0
            R0_est = peak_prev / I[0]
            Re_t_array = R0_est * S / N
            below_1_indices = np.where(Re_t_array < 1)[0]
            time_Re_below_1 = time[below_1_indices[0]] if len(below_1_indices) > 0 else np.nan
        except:
            doubling_time = np.nan
            time_Re_below_1 = np.nan
    
    results_clean['epidemic_duration'].append(epidemic_duration)
    results_clean['peak_prevalence'].append(peak_prev)
    results_clean['peak_time'].append(peak_time)
    results_clean['final_epidemic_size'].append(final_size)
    results_clean['doubling_time'].append(doubling_time)
    results_clean['time_Re_below_1'].append(time_Re_below_1)
    results_clean['initial_S'].append(init_S)
    results_clean['final_S'].append(final_S)
    results_clean['outbreak'].append(outbreak)

res_df_clean = pd.DataFrame(results_clean)

metrics_clean = {}

metrics_clean['epidemic_duration_mean'] = np.nanmean(res_df_clean['epidemic_duration'])
metrics_clean['epidemic_duration_median'] = np.nanmedian(res_df_clean['epidemic_duration'])
metrics_clean['peak_prevalence_mean'] = np.nanmean(res_df_clean['peak_prevalence'])
metrics_clean['peak_prevalence_median'] = np.nanmedian(res_df_clean['peak_prevalence'])
metrics_clean['peak_time_mean'] = np.nanmean(res_df_clean['peak_time'])
metrics_clean['peak_time_median'] = np.nanmedian(res_df_clean['peak_time'])
metrics_clean['final_epidemic_size_mean'] = np.nanmean(res_df_clean['final_epidemic_size'])
metrics_clean['final_epidemic_size_median'] = np.nanmedian(res_df_clean['final_epidemic_size'])
metrics_clean['doubling_time_mean'] = np.nanmean(res_df_clean['doubling_time'])
metrics_clean['doubling_time_median'] = np.nanmedian(res_df_clean['doubling_time'])
metrics_clean['time_Re_below_1_mean'] = np.nanmean(res_df_clean['time_Re_below_1'])
metrics_clean['time_Re_below_1_median'] = np.nanmedian(res_df_clean['time_Re_below_1'])
metrics_clean['initial_S_mean'] = np.nanmean(res_df_clean['initial_S'])
metrics_clean['initial_S_median'] = np.nanmedian(res_df_clean['initial_S'])
metrics_clean['final_S_mean'] = np.nanmean(res_df_clean['final_S'])
metrics_clean['final_S_median'] = np.nanmedian(res_df_clean['final_S'])
metrics_clean['outbreak_probability'] = np.nanmean(res_df_clean['outbreak'])

metrics_clean