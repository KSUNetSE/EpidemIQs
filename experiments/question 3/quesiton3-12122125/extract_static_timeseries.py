
# Extract mean and 90% CI time series for S, I, R from the static data
# Note: to reduce size, I'll sample or aggregate time points if needed
# For now, just take as is

# Rename columns for clarity
static_time_series = results_static.rename(columns={
    'S': 'S_mean',
    'S_90ci_lower': 'S_90ci_lower',
    'S_90ci_upper': 'S_90ci_upper',
    'I': 'I_mean',
    'I_90ci_lower': 'I_90ci_lower',
    'I_90ci_upper': 'I_90ci_upper',
    'R': 'R_mean',
    'R_90ci_lower': 'R_90ci_lower',
    'R_90ci_upper': 'R_90ci_upper'
})

static_time_series.head()