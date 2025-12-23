
import numpy as np
from scipy import stats

systemic_int_22 = df22['systemic'].astype(int)

prob_systemic_22 = systemic_int_22.mean() * 100

mean_cascade_22 = df22['n_final_failed'].mean()
median_cascade_22 = df22['n_final_failed'].median()
std_cascade_22 = df22['n_final_failed'].std()
mode_res_22 = stats.mode(df22['n_final_failed'])
mode_cascade_22 = mode_res_22.mode.item()

mean_rounds_22 = df22['rounds'].mean()
median_rounds_22 = df22['rounds'].median()

(prob_systemic_22, mean_cascade_22, median_cascade_22, std_cascade_22, mode_cascade_22, mean_rounds_22, median_rounds_22)