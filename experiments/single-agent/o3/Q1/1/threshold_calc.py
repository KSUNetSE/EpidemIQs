
import numpy as np
mean_k_er = 7.9756
sec_er = 71.49
mean_k_ba = 7.9936
sec_ba = 167.1968
beta_over_gamma_thr_er = mean_k_er / sec_er
beta_over_gamma_thr_ba = mean_k_ba / sec_ba
beta_over_gamma_homogeneous=1
beta_over_gamma_er=1/beta_over_gamma_thr_er
beta_over_gamma_ba=1/beta_over_gamma_thr_ba
result={'beta_over_gamma_threshold_homogeneous':1.0,'beta_over_gamma_threshold_er':beta_over_gamma_thr_er,'beta_over_gamma_threshold_ba':beta_over_gamma_thr_ba,'factor_heterogeneity_increase_er':1/beta_over_gamma_thr_er,'factor_heterogeneity_increase_ba':1/beta_over_gamma_thr_ba}
