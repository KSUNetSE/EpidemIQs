
# Step 5: Compute beta for desired R0 on SBM network, SEIBR dynamics
# For SEIBR, T = beta/(beta + gamma1), and for SBM: R0 = T * ((k2_mean - k_mean) / k_mean)

R0 = 8.0
sigma = 2.0  # latent rate (E->I)
gamma1 = 0.7 # infectious period (I->B)
gamma2 = 0.5 # bed-confined period (B->R)
k_mean = 22.82
k2_mean = 542.66
D = (k2_mean - k_mean) / k_mean

# We want to solve for beta in: R0 = (beta / (beta + gamma1)) * D
# R0 = [beta/(beta + gamma1)] * D =>
# R0/D = beta/(beta + gamma1)
# (R0/D)*(beta + gamma1) = beta
# (R0/D)*beta + (R0/D)*gamma1 = beta
# ((R0/D)*gamma1) = beta - (R0/D)*beta
# ((R0/D)*gamma1) = beta * (1 - (R0/D))
# beta = ((R0/D)*gamma1) / (1 - (R0/D))

Dval = D
R0_D = R0 / Dval

if R0_D >= 1.0:
    raise ValueError("Final-size or network clustering incompatible with R0: denominator can go negative or infinite.")

beta = (R0_D * gamma1) / (1 - R0_D)

# Set as dict with all rates per CTMC
params = {
    "beta": round(beta, 4),
    "sigma": round(sigma, 4),
    "gamma1": round(gamma1, 4),
    "gamma2": round(gamma2, 4)
}

# Step 7: Initial conditions in percent for N=763, seed as E(0)=1
N = 763
S0 = N-1
E0 = 1
I0 = 0
B0 = 0
R0pop = 0
perc_S = round((S0 / N) * 100)
perc_E = round((E0 / N) * 100)
perc_I = 0
perc_B = 0
perc_R = 0
# adjust sum to be exactly 100 (rounding error; assign the loss to S)
over = perc_S + perc_E + perc_I + perc_B + perc_R - 100
perc_S -= over
init_cond = [{"S": perc_S, "E": perc_E, "I": perc_I, "B": perc_B, "R": perc_R}]

(params, init_cond)