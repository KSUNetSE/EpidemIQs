
import numpy as np
from scipy import stats

systemic_int_31 = df31['systemic'].astype(int)

prob_systemic_31 = systemic_int_31.mean() * 100

mean_cascade_31 = df31['n_final_failed'].mean()
median_cascade_31 = df31['n_final_failed'].median()
std_cascade_31 = df31['n_final_failed'].std()
mode_res_31 = stats.mode(df31['n_final_failed'])
mode_cascade_31 = mode_res_31.mode.item()

mean_rounds_31 = df31['rounds'].mean()
median_rounds_31 = df31['rounds'].median()

(prob_systemic_31, mean_cascade_31, median_cascade_31, std_cascade_31, mode_cascade_31, mean_rounds_31, median_rounds_31)