
# Chain-of-Thought Reasoning (Iteration 1):
# Step 1: ODE SIR Baseline (Well-mixed model, R0=3, analytical scenario) [Scenario 1, Model#0]
# - No network required. Analytical/deterministic integration of standard ODE SIR model.
# - Simulate for N=1000, beta=0.3, gamma=0.1, initial [S=999, I=1, R=0], for 300 days.
# - Save output for timecourse and plot as required format (results-10.csv, results-10.png).
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import matplotlib.pyplot as plt
import os

N = 1000
beta = 0.3
gamma = 0.1

# Initial conditions
S0, I0, R0 = 999, 1, 0
IC = [S0, I0, R0]

def SIR_ODE(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

tspan = np.linspace(0, 300, 501)
sol = odeint(SIR_ODE, IC, tspan, args=(beta, gamma, N))
S, I, R = sol.T

# Save results as CSV
outdir = os.path.join(os.getcwd(), 'output')
os.makedirs(outdir, exist_ok=True)
results_10_csv = os.path.join(outdir, 'results-10.csv')
df = pd.DataFrame({'time': tspan, 'S': S, 'I': I, 'R': R})
df.to_csv(results_10_csv, index=False)

# Plot results
results_10_png = os.path.join(outdir, 'results-10.png')
plt.figure(figsize=(8,5))
plt.plot(tspan, S, label='Susceptible')
plt.plot(tspan, I, label='Infectious')
plt.plot(tspan, R, label='Removed')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.legend()
plt.title('ODE SIR: Well-mixed, R0=3, N=1000')
plt.tight_layout()
plt.savefig(results_10_png)
plt.close()

# Return paths for downstream bookkeeping
(results_10_csv, results_10_png)
