
# The 'peak_B' returned is a series which is not right, we only need max value
peak_b_metrics_corrected = {
    'peak_B': float(results_11['B'].max()),  # maximum prevalence value
    'peak_time_B': float(results_11.loc[results_11['B'].idxmax(), 'time']),
    'duration_B': float(epidemic_duration),
    'B_90ci_width_at_peak': float(b_90ci_width_at_peak)
}
peak_b_metrics_corrected