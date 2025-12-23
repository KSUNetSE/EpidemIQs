
# Purpose: Analytical demonstration of why the chain of transmission breaks in SIR, due to decline in infectives, not exhaustion of susceptibles
import numpy as np
import matplotlib.pyplot as plt

def SIR_analytical(beta, gamma, S0, I0, R0, N, T):
    S, I, R = np.zeros(T), np.zeros(T), np.zeros(T)
    S[0], I[0], R[0] = S0, I0, R0
    for t in range(1, T):
        new_infections = beta * S[t-1] * I[t-1]/N
        new_recoveries = gamma * I[t-1]
        S[t] = S[t-1] - new_infections
        I[t] = I[t-1] + new_infections - new_recoveries
        R[t] = R[t-1] + new_recoveries
    return S, I, R

# Parameters
beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
R0_ = 0
S0 = N - I0 - R0_
T = 200

S, I, R = SIR_analytical(beta, gamma, S0, I0, R0_, N, T)

plt.figure(figsize=(8,5))
plt.plot(S, label='Susceptibles')
plt.plot(I, label='Infectives')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.title('SIR Model: Chain of Transmission Break')
plt.legend()
plt.tight_layout()
plt.savefig('output/results-10.png')

# The key test: at the end, is S(T)>0 but I(T) ~ 0?
end_S = S[-1]
end_I = I[-1]

(end_S, end_I)