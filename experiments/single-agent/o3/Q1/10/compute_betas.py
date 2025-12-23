
import numpy as np
mean_k_er = 10.0376
second_moment_er = 110.8
mean_k_ba = 9.99
second_moment_ba = 272.5716
R0=2.5
gamma=0.2
beta_er = R0*gamma*mean_k_er/(second_moment_er-mean_k_er)
beta_ba = R0*gamma*mean_k_ba/(second_moment_ba-mean_k_ba)
return_dict={'beta_er':beta_er,'beta_ba':beta_ba}
