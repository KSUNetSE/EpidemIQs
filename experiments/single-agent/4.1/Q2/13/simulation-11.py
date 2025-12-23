
# Modeling phase
# Analytical exploration: SIR model equations - two scenarios
# 1. Epidemic decline from recovery (I falls to near zero) 2. Epidemic stalls from lack of susceptibles (S goes to zero)

import numpy as np
import matplotlib.pyplot as plt

# SIR model on mean-field network (analytical population level)
N = 10000 # population size
I0, R0 = 10, 0
beta = 0.3
gamma = 0.1
S0 = N - I0 - R0


def sir_sim(S0, I0, R0, beta, gamma, T, dt):
    t_vals = np.arange(0, T, dt)
    S = np.zeros(len(t_vals))
    I = np.zeros(len(t_vals))
    R = np.zeros(len(t_vals))
    S[0], I[0], R[0] = S0, I0, R0
    N = S0 + I0 + R0
    for t in range(1, len(t_vals)):
        dS = -beta * S[t-1] * I[t-1] / N
        dI = beta * S[t-1] * I[t-1] / N - gamma * I[t-1]
        dR = gamma * I[t-1]
        S[t] = S[t-1] + dS*dt
        I[t] = I[t-1] + dI*dt
        R[t] = R[t-1] + dR*dt
    return t_vals, S, I, R

# Case 1: Normal epidemic - decline due to lack of susceptibles
T, dt = 160, 0.5
t1, S1, I1, R1 = sir_sim(S0, I0, R0, beta, gamma, T, dt)

# Case 2: Start with zero susceptibles (S=0) -- artificial, epidemic dies instantly
I0_case2 = 10
S0_case2 = 0
R0_case2 = N - I0_case2

t2, S2, I2, R2 = sir_sim(S0_case2, I0_case2, R0_case2, beta, gamma, T, dt)

plt.figure(figsize=(10,5))
plt.plot(t1, S1, label='S, normal')
plt.plot(t1, I1, label='I, normal')
plt.plot(t1, R1, label='R, normal')
plt.plot(t2, I2, '--', label='I, S=0 case')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR model: transmission breaks by decline in I vs lack of susceptibles')
plt.legend()
plt.tight_layout()
plt.savefig('output/results-11.png')

import pandas as pd
data = pd.DataFrame({'t': t1, 'S': S1, 'I': I1, 'R': R1})
data.to_csv('output/results-11.csv', index=False)
# Also save short S=0 analytical result
data2 = pd.DataFrame({'t': t2, 'I_noS': I2})
data2.to_csv('output/results-12.csv', index=False)
