
# Analysis of deterministic SEIR ODEs for homogeneous-mixing (mean-field)
from scipy.integrate import odeint
import numpy as np
import pandas as pd
# Parameters
N = 1000
sigma = 1/3
gamma = 1/5
beta = 0.04115  # From previous matching for R0~2.5 (mean degree 12)
IC = [0.98*N, 0, 0.01*N, 0.01*N]  # S,E,I,R
Tmax = 180

def seir_deriv(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I /(N)
    dEdt = beta * S * I /(N) - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

t = np.linspace(0, Tmax, 360)  # 0.5 day steps
y = odeint(seir_deriv, IC, t, args=(N, beta, sigma, gamma))
pd.DataFrame({'time': t, 'S': y[:,0], 'E':y[:,1], 'I':y[:,2], 'R':y[:,3]}).to_csv(os.path.join(os.getcwd(), 'output', 'results-1-3.csv'), index=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,4))
plt.plot(t, y[:,0], label='S')
plt.plot(t, y[:,1], label='E')
plt.plot(t, y[:,2], label='I')
plt.plot(t, y[:,3], label='R')
plt.legend()
plt.title('Deterministic SEIR (ODE, homogeneous-mixing)\nInitial: 1% I, 1% R')
plt.xlabel('Time (days)')
plt.ylabel('Number')
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-1-3.png'))
plt.close()