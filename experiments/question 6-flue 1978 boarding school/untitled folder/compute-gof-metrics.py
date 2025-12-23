
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import interp1d
import numpy as np

# Interpolate simulation B and C to empirical day points
interp_B = interp1d(sim_time, sim_B, kind='linear', fill_value='extrapolate')
interp_C = interp1d(sim_time, sim_C, kind='linear', fill_value='extrapolate')

sim_B_interp = interp_B(empirical_days)
sim_C_interp = interp_C(empirical_days)

# Calculate goodness-of-fit metrics for B
rmse_B = np.sqrt(mean_squared_error(empirical_B, sim_B_interp))
mae_B = mean_absolute_error(empirical_B, sim_B_interp)
r2_B = r2_score(empirical_B, sim_B_interp)

# Calculate goodness-of-fit metrics for C
rmse_C = np.sqrt(mean_squared_error(empirical_C, sim_C_interp))
mae_C = mean_absolute_error(empirical_C, sim_C_interp)
r2_C = r2_score(empirical_C, sim_C_interp)

# Return all metrics and interpolated series length
rmse_B, mae_B, r2_B, rmse_C, mae_C, r2_C, len(empirical_days)