
# Reason: Implement scenario (i=1, j=3): Deterministic mean-field ODE SEIR simulation (mass-action/"well-mixed"") for direct comparison with the network models.
# Output: results-13.csv (.png)
import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Network size identical to previous network sims
N = 2000
# Use mean degree for ER = 8.07 -> well-mixed population beta = beta_edge * mean_k
mean_k_er = 8.07
beta = 0.1363 * mean_k_er
sigma = 0.2
gamma = 0.43

# Initial conditions: 1996 S, 3 E, 1 I, 0 R
S0, E0, I0, R0 = 1996, 3, 1, 0
Y0 = [S0, E0, I0, R0]

def SEIR_ode(t, y):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

T_end = 120  # days
sol = solve_ivp(SEIR_ode, [0, T_end], Y0, t_eval=np.linspace(0, T_end, 901), method='RK45')

ode_results = pd.DataFrame({
    'time': sol.t,
    'S': sol.y[0],
    'E': sol.y[1],
    'I': sol.y[2],
    'R': sol.y[3],
})
csv_path = os.path.join(os.getcwd(), 'output', 'results-13.csv')
ode_results.to_csv(csv_path, index=False)

# Plot
png_path = os.path.join(os.getcwd(), 'output', 'results-13.png')
plt.figure(figsize=(7,5))
plt.plot(sol.t, sol.y[0], label='S', color='tab:blue')
plt.plot(sol.t, sol.y[1], label='E', color='tab:orange')
plt.plot(sol.t, sol.y[2], label='I', color='tab:red')
plt.plot(sol.t, sol.y[3], label='R', color='tab:green')
plt.xlabel('Time (days)')
plt.ylabel('Count')
plt.title('Deterministic SEIR ODE (well-mixed, ER equivalent)')
plt.legend()
plt.tight_layout()
plt.savefig(png_path)
plt.close()
csv_path, png_path, beta, sigma, gamma