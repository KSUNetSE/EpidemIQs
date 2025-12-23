
# Extract empirical data
# From results_14, B_obs is bed-confined cases, C_obs is cumulative cases

empirical_df = results_14.copy()

# Rename for clarity
empirical_df.rename(columns={'day': 'day', 'B_obs': 'B_emp', 'C_obs': 'R_emp'}, inplace=True)

# Display empirical data head
empirical_df.head()