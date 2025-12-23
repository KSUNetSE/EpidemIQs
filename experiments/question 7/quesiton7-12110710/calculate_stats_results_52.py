
import numpy as np
from scipy import stats

systemic_int_52 = df52['systemic'].astype(int)

prob_systemic_52 = systemic_int_52.mean() * 100

mean_cascade_52 = df52['n_final_failed'].mean()
median_cascade_52 = df52['n_final_failed'].median()
std_cascade_52 = df52['n_final_failed'].std()
mode_res_52 = stats.mode(df52['n_final_failed'])
mode_cascade_52 = mode_res_52.mode.item()

mean_rounds_52 = df52['rounds'].mean()
median_rounds_52 = df52['rounds'].median()

(prob_systemic_52, mean_cascade_52, median_cascade_52, std_cascade_52, mode_cascade_52, mean_rounds_52, median_rounds_52)