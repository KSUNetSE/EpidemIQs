
# Initial condition for N=10000, I0 = 10 (randomly chosen), S0 = 9990, R0 = 0
N = 10000
I0 = 10
S0 = N - I0
R0_ = 0
init_cond = {'S': int(round(S0*100/N)), 'I': int(round(I0*100/N)), 'R': 0}
# Adjust to ensure sum = 100
remainder = 100 - (init_cond['S'] + init_cond['I'] + init_cond['R'])
if remainder != 0:
    init_cond['S'] += remainder
initial_conditions = [init_cond]
initial_condition_desc = ["10 randomly chosen nodes initially infectious (0.1%), 9990 susceptible (99.9%), 0 recovered; applies to temporal and static networks."]
initial_conditions, initial_condition_desc