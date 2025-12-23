
import numpy as np
# Activity-driven temporal network parameters
N = 1000  # nodes
alpha = 0.1  # node activation probability per time step (activity rate)
m = 2  # links per activation

# Compute <a> and <a^2> for homogeneous activity (all have alpha), as in this case
a_vals = np.ones(N) * alpha
activity_mean = np.mean(a_vals)
activity_sq_mean = np.mean(a_vals**2)

# Temporal network threshold (see e.g. PRL 109, 118702 (2012); beta/gamma > 1 / [m ( <a> + sqrt(<a^2>) ) ])
epsilon = 1e-9
beta_gamma_threshold_activity = 1 / (m * (activity_mean + np.sqrt(activity_sq_mean)))

# For the static time-aggregated network (all contacts summed):
# Each time step, about N*alpha*m edges are formed, edges last one step, so over T steps,
# the aggregated degree per node k_agg ~ alpha * m * T
k_mean = alpha * m * 1  # per unit time
k_sq_mean = k_mean**2  # Since homogeneous, all same degree

beta_gamma_threshold_aggregated = k_mean / (k_sq_mean + epsilon)  # threshold for static random regular graph (bond percolation)

{
  'activity_mean': activity_mean,
  'activity_sq_mean': activity_sq_mean,
  'beta_gamma_threshold_activity': beta_gamma_threshold_activity,
  'beta_gamma_threshold_aggregated': beta_gamma_threshold_aggregated
}