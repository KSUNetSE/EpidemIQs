
# Step 1: Import needed modules
import math

# Step 2: Network and Epidemic Parameters
N = 1000                      # Number of nodes
m = 1                         # Per activation: number of contacts (given)
mean_a = 0.0163               # <a>, mean activity, from diagnostics
mean_a2 = 0.00041             # <a^2>, second moment, from diagnostics
R0 = 3.0                      # Target basic reproduction number

# Step 3: Compute denominator for R0 equation in activity-driven network
sqrt_mean_a2 = math.sqrt(mean_a2)
denominator = m * (mean_a + sqrt_mean_a2)

# Step 4: β/μ computation
beta_over_mu = R0 / denominator

# Step 5: Fix μ (recovery rate)
mu = 1.0                      # mean infectious period = 1 (arbitrary time unit)

# Step 6: Compute β numerically
beta = beta_over_mu * mu

# Format for output
sir_params = {"beta": round(beta, 5), "mu": round(mu, 5)}

sir_params