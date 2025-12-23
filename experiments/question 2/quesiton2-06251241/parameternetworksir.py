
# Step 1: Network statistics from ER and BA networks
k_mean_er = 10.022  # ER: <k>
k2_mean_er = 110.4  # ER: <k^2>
k_mean_ba = 9.95    # BA: <k>
k2_mean_ba = 205.5  # BA: <k^2>

# Step 2: Compute mean excess degree q for each
q_er = (k2_mean_er - k_mean_er) / k_mean_er
q_ba = (k2_mean_ba - k_mean_ba) / k_mean_ba

# Step 3: Set R_0 and gamma for both scenarios (R_0 > 1, R_0 < 1)
R0_hi = 3.0
R0_lo = 0.5
gamma = 0.1

# Step 4: Compute beta per-contact for each network and scenario
# beta_network = R0 * gamma / q

def get_beta(R0, gamma, q):
    return R0 * gamma / q

beta_er_hi = get_beta(R0_hi, gamma, q_er)
beta_er_lo = get_beta(R0_lo, gamma, q_er)
beta_ba_hi = get_beta(R0_hi, gamma, q_ba)
beta_ba_lo = get_beta(R0_lo, gamma, q_ba)

# Step 5: Initial conditions (percentages) from prompt:
# S(0) = 990, I(0) = 10, R(0) = 0, N=1000 --> 99% S, 1% I, 0% R
init_cond = {"S": 99, "I": 1, "R": 0}

# Step 6: Prepare output dicts for all 4 combinations
params_all = {
    "ER_R0_gt1": {"beta": beta_er_hi, "gamma": gamma},
    "ER_R0_lt1": {"beta": beta_er_lo, "gamma": gamma},
    "BA_R0_gt1": {"beta": beta_ba_hi, "gamma": gamma},
    "BA_R0_lt1": {"beta": beta_ba_lo, "gamma": gamma},
}

q_values = {"q_er": q_er, "q_ba": q_ba}

params_all, q_values, init_cond