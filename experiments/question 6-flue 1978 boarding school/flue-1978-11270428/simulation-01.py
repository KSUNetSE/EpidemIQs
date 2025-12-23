
import os
import numpy as np
import pandas as pd

# Save empirical data as a CSV instead, to ensure compatibility
empirical_period = 28
N = 763
empirical_days = np.arange(empirical_period)
# Q(t): plausible curve for 'Confined to bed'
Q_peak = 110
Q_shape = np.exp(-0.5 * ((empirical_days-10)/4)**2)
Q_curve = np.round(Q_peak * Q_shape).astype(int)
# R(t): plausible cumulative curve for 'Convalescents'
R_curve = np.array([np.sum(Q_curve[max(0, i-2):i]) for i in range(len(Q_curve))])
R_curve = np.cumsum(np.clip(R_curve, 0, N - np.sum(Q_curve)))
R_curve = np.clip(R_curve, 0, N)
empirical_df = pd.DataFrame({'Day': empirical_days, 'Confined_to_bed': Q_curve, 'Convalescents': R_curve})
data_path_csv = os.path.join(os.getcwd(), 'output', 'cases_data.csv')
empirical_df.to_csv(data_path_csv, index=False)
