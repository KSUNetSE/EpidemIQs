
# Recreate daily aggregated timeseries for results_11 for error calculations
results_11['day'] = results_11['time'].round().astype(int)
daily_11 = results_11.groupby('day').agg({'B': 'mean'}).reset_index()

# Use empirical_df from earlier extraction

# Calculate RMSE and MAE between daily_11 B and empirical B_emp
from sklearn.metrics import mean_squared_error, mean_absolute_error

sim_B = daily_11.set_index('day')['B']
emp_B = empirical_df.set_index('day')['B_emp']
combined = pd.concat([sim_B, emp_B], axis=1, join='outer').fillna(0)
rmse_11 = np.sqrt(mean_squared_error(combined['B_emp'], combined['B']))
mae_11 = mean_absolute_error(combined['B_emp'], combined['B'])

rmse_11, mae_11