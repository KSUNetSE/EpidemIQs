
# Optional: Mean-field SIRS ODE for comparison (i=2, j=1)
# We use the same parameters as in the main simulation. N = 1000.
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

def sirs_upf_ode(y, t, beta, gamma, xi):
    U, P, F = y
    N = U + P + F
    # Network-compensated force-of-infection: <k>*P/N, mean field
    dUdt = -beta * U * P / N + xi * F
    dPdt = beta * U * P / N - gamma * P
    dFdt = gamma * P - xi * F
    return [dUdt, dPdt, dFdt]

N = 1000
U0, P0, F0 = int(0.99*N), int(0.01*N), 0
y0 = [U0, P0, F0]      # Initial state counts
t = np.linspace(0, 400, 801)  # 0 to 400 days, 0.5d steps

beta = 0.8
gamma = 0.3
xi = 0.1

sol = odeint(sirs_upf_ode, y0, t, args=(beta, gamma, xi))
U, P, F = sol.T

# Save all results
data = pd.DataFrame({'time': t, 'U': U, 'P': P, 'F': F, 'U_prop': U/N, 'P_prop': P/N, 'F_prop': F/N})
csv_path = os.path.join(os.getcwd(), 'output', 'results-21.csv')
data.to_csv(csv_path, index=False)

# Plot: absolute number and proportion (proportion primary)
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(t, U/N, label='Unaware (U)', color='tab:blue')
ax.plot(t, P/N, label='Posting (P)', color='tab:red')
ax.plot(t, F/N, label='Fatigued (F)', color='tab:gray')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Proportion of population')
ax.set_title('Mean-field SIRS-UPF (ODE) baseline')
ax.legend()
plt.tight_layout()
png_path = os.path.join(os.getcwd(), 'output', 'results-21.png')
plt.savefig(png_path, dpi=180)
plt.close()
