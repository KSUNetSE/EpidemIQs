
import numpy as np
from scipy import stats

systemic_int_51 = df51['systemic'].astype(int)

prob_systemic_51 = systemic_int_51.mean() * 100

mean_cascade_51 = df51['n_final_failed'].mean()
median_cascade_51 = df51['n_final_failed'].median()
std_cascade_51 = df51['n_final_failed'].std()
mode_res_51 = stats.mode(df51['n_final_failed'])
mode_cascade_51 = mode_res_51.mode.item()

mean_rounds_51 = df51['rounds'].mean()
median_rounds_51 = df51['rounds'].median()

(prob_systemic_51, mean_cascade_51, median_cascade_51, std_cascade_51, mode_cascade_51, mean_rounds_51, median_rounds_51)