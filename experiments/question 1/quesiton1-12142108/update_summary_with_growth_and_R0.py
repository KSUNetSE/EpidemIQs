
# Add calculated growth rate metrics to the summary dataframe
summary_df.loc[summary_df['Metric'] == 'Early growth rate r', 'Value'] = r
summary_df.loc[summary_df['Metric'] == 'Doubling time (days)', 'Value'] = doubling_time
summary_df.loc[summary_df['Metric'] == 'Theoretical R0', 'Value'] = R0_theoretical

# Re-save the updated summary
summary_df.to_csv(csv_path, index=False)
csv_path