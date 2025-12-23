
def get_daily_B(df):
    df['day'] = df['time'].round().astype(int)
    daily = df.groupby('day')['B'].mean().reset_index()
    return daily

errors = {}
for key, df in [('results_12', results_12), ('results_13', results_13), ('results_16', results_16)]:
    daily_sim = get_daily_B(df)
    sim_B = daily_sim.set_index('day')['B']
    emp_B = empirical_df.set_index('day')['B_emp']
    combined = pd.concat([sim_B, emp_B], axis=1, join='outer').fillna(0)
    rmse_val = np.sqrt(mean_squared_error(combined['B_emp'], combined['B']))
    mae_val = mean_absolute_error(combined['B_emp'], combined['B'])
    errors[key] = {'RMSE': rmse_val, 'MAE': mae_val}

errors