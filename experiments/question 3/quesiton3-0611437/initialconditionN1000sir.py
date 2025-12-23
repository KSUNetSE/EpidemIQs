
# Initial conditions translation for N=1000
N = 1000
initial_I = int(round(0.01 * N))     # 1% infected
initial_R = 0
initial_S = N - initial_I - initial_R
init_cond = {'S': initial_S * 100 // N, 'I': initial_I * 100 // N, 'R': 0}
# Ensures S, I, R sum to 100 exactly
if sum(init_cond.values()) < 100:
    init_cond['S'] += 100 - sum(init_cond.values())
init_cond