
import numpy as np
from scipy import stats

systemic_int_32 = df32['systemic'].astype(int)

prob_systemic_32 = systemic_int_32.mean() * 100

mean_cascade_32 = df32['n_final_failed'].mean()
median_cascade_32 = df32['n_final_failed'].median()
std_cascade_32 = df32['n_final_failed'].std()
mode_res_32 = stats.mode(df32['n_final_failed'])
mode_cascade_32 = mode_res_32.mode.item()

mean_rounds_32 = df32['rounds'].mean()
median_rounds_32 = df32['rounds'].median()

(prob_systemic_32, mean_cascade_32, median_cascade_32, std_cascade_32, mode_cascade_32, mean_rounds_32, median_rounds_32)