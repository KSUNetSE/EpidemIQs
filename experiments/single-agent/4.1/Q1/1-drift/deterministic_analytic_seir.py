
# Analytic deterministic ODE for homogeneous mixing SEIR (population fractions)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

N = 2000
Tmax = 120
beta = 0.46667
sigma = 0.33333
gamma = 0.16667
init_I = 40
init_E = 0
init_R = 0
init_S = N - init_I

steps = Tmax+1
t = np.linspace(0, Tmax, steps)
S = np.zeros(steps)
E = np.zeros(steps)
I = np.zeros(steps)
R = np.zeros(steps)
S[0] = init_S / N
E[0] = init_E / N
I[0] = init_I / N
R[0] = init_R / N

for i in range(steps-1):
    dS = -beta * S[i] * I[i]
    dE = beta * S[i] * I[i] - sigma * E[i]
    dI = sigma * E[i] - gamma * I[i]
    dR = gamma * I[i]
    S[i+1] = S[i] + dS
    E[i+1] = E[i] + dE
    I[i+1] = I[i] + dI
    R[i+1] = R[i] + dR

plt.figure(figsize=(6,4))
plt.plot(t, S*N, label='S')
plt.plot(t, E*N, label='E')
plt.plot(t, I*N, label='I')
plt.plot(t, R*N, label='R')
plt.xlabel('Day')
plt.ylabel('Population')
plt.legend()
plt.title('Deterministic SEIR - Homogeneous Mixing')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-13.png'))
pd.DataFrame({'time': t, 'S': S*N, 'E':E*N, 'I':I*N, 'R':R*N}).to_csv(os.path.join(os.getcwd(), 'output', 'results-13.csv'), index=False)