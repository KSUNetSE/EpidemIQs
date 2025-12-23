
import numpy as np
from scipy import stats

systemic_int_42 = df42['systemic'].astype(int)

prob_systemic_42 = systemic_int_42.mean() * 100

mean_cascade_42 = df42['n_final_failed'].mean()
median_cascade_42 = df42['n_final_failed'].median()
std_cascade_42 = df42['n_final_failed'].std()
mode_res_42 = stats.mode(df42['n_final_failed'])
mode_cascade_42 = mode_res_42.mode.item()

mean_rounds_42 = df42['rounds'].mean()
median_rounds_42 = df42['rounds'].median()

(prob_systemic_42, mean_cascade_42, median_cascade_42, std_cascade_42, mode_cascade_42, mean_rounds_42, median_rounds_42)