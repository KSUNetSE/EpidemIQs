
import numpy as np
from scipy import stats

systemic_int_62 = df62['systemic'].astype(int)

prob_systemic_62 = systemic_int_62.mean() * 100

mean_cascade_62 = df62['n_final_failed'].mean()
median_cascade_62 = df62['n_final_failed'].median()
std_cascade_62 = df62['n_final_failed'].std()
mode_res_62 = stats.mode(df62['n_final_failed'])
mode_cascade_62 = mode_res_62.mode.item()

mean_rounds_62 = df62['rounds'].mean()
median_rounds_62 = df62['rounds'].median()

(prob_systemic_62, mean_cascade_62, median_cascade_62, std_cascade_62, mode_cascade_62, mean_rounds_62, median_rounds_62)