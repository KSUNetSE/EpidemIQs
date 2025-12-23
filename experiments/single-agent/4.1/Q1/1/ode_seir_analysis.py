
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- Classic mean-field/ODE homogeneous-mixing SEIR model ---
N = 1000
R0 = 2.5
D_latent = 3.0
D_inf = 4.0
sigma = 1.0 / D_latent
gamma = 1.0 / D_inf
beta = R0 * gamma  # per-individual, matches ODE scale

def seir_ode(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Initial: 5 I, 0 E, 995 S, 0 R
init = [995, 0, 5, 0]
t = np.linspace(0, 120, 1200)
sol = odeint(seir_ode, init, t, args=(N, beta, sigma, gamma))

# Save for paper
df = pd.DataFrame(sol, columns=["S", "E", "I", "R"])
df["time"] = t
df.to_csv(os.path.join(os.getcwd(), 'output', 'results-ode-seir.csv'), index=False)

# Plot
plt.figure(figsize=(8,5))
plt.plot(t, sol[:,2], label='I (ODE)')
plt.plot(t, sol[:,3], label='R (ODE)')
plt.xlabel('Time')
plt.ylabel('Individuals')
plt.title('SEIR ODE (Homogeneous Mixing)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-ode-seir.png'))
plt.close()

# ODE metrics: peak I, time to peak, final size, duration (I<1)
duration = t[np.argmax(sol[:,2]<1)]
metrics_ode = {
    'final_size': sol[-1,3],
    'peak_I': np.max(sol[:,2]),
    'time_to_peak': t[np.argmax(sol[:,2])],
    'duration': duration
}
metrics_ode