
# Check min/max and sums to estimate scaling
max_S = susceptible.max()
min_S = susceptible.min()
max_I = infected.max()
min_I = infected.min()
max_R = recovered.max()
min_R = recovered.min()
sum_SIR = susceptible + infected + recovered
constant_sum = (np.allclose(sum_SIR, sum_SIR[0]))  # is sum constant?
sum_value = sum_SIR[0]

max_S, min_S, max_I, min_I, max_R, min_R, constant_sum, sum_value