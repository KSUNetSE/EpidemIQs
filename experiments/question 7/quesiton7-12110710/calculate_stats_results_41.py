
import numpy as np
from scipy import stats

systemic_int_41 = df41['systemic'].astype(int)

prob_systemic_41 = systemic_int_41.mean() * 100

mean_cascade_41 = df41['n_final_failed'].mean()
median_cascade_41 = df41['n_final_failed'].median()
std_cascade_41 = df41['n_final_failed'].std()
mode_res_41 = stats.mode(df41['n_final_failed'])
mode_cascade_41 = mode_res_41.mode.item()

mean_rounds_41 = df41['rounds'].mean()
median_rounds_41 = df41['rounds'].median()

(prob_systemic_41, mean_cascade_41, median_cascade_41, std_cascade_41, mode_cascade_41, mean_rounds_41, median_rounds_41)