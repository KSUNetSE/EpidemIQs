
# Calculate descriptive statistics and check for NaNs in each metric column
t_10_stats = data['t_10'].describe()
t_50_stats = data['t_50'].describe()
I_peak_stats = data['I_peak'].describe()

# Count of NaNs per column
t_10_nans = data['t_10'].isna().sum()
t_50_nans = data['t_50'].isna().sum()
I_peak_nans = data['I_peak'].isna().sum()

# Prepare summary
summary_stats = {
    't_10_stats': t_10_stats.to_dict(),
    't_50_stats': t_50_stats.to_dict(),
    'I_peak_stats': I_peak_stats.to_dict(),
    't_10_nans': t_10_nans,
    't_50_nans': t_50_nans,
    'I_peak_nans': I_peak_nans
}