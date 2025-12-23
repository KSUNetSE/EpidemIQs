
import numpy as np
mean_k = 2.9068
k2 = 16.6278
r=0.1125
q_orig = 4.72031
beta_over_gamma = 4/q_orig
# solve q_new condition q_new< 1/beta_over_gamma ~? Wait: R0_new = beta_gamma * q_new; need <1 => q_new < 1/(beta/gamma)
# compute
threshold = 1/ (beta_over_gamma)
threshold
