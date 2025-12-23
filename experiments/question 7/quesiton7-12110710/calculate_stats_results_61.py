
import numpy as np
from scipy import stats

systemic_int_61 = df61['systemic'].astype(int)

prob_systemic_61 = systemic_int_61.mean() * 100

mean_cascade_61 = df61['n_final_failed'].mean()
median_cascade_61 = df61['n_final_failed'].median()
std_cascade_61 = df61['n_final_failed'].std()
mode_res_61 = stats.mode(df61['n_final_failed'])
mode_cascade_61 = mode_res_61.mode.item()

mean_rounds_61 = df61['rounds'].mean()
median_rounds_61 = df61['rounds'].median()

(prob_systemic_61, mean_cascade_61, median_cascade_61, std_cascade_61, mode_cascade_61, mean_rounds_61, median_rounds_61)