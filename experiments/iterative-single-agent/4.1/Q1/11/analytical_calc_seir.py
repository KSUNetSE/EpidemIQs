
# Analytical calculation of epidemic threshold (R0) for SEIR on both networks
# For configuration/network-based SEIR, threshold can often be written as: R0 = (beta/gamma) * (⟨k^2⟩-⟨k⟩)/⟨k⟩, but latent/exposed period modifies dynamics somewhat.
# For analytical purposes we will use the major (S,L,E,I,R) compartment model with transitions:
# S -(I)-> E (rate beta)
# E -> I (rate sigma)
# I -> R (rate gamma)
# This yields a deterministic ODE for both homogeneous and heterogeneous mixing, but analytic epidemic threshold on a network is linked to largest eigenvalue of network or to (⟨k^2⟩-⟨k⟩)/⟨k⟩

# Parameters:
beta = 0.07     # Transmission rate per contact per unit time (chosen to get R0>1 for study)
sigma = 0.2     # Rate from exposed to infectious (1/latent period=5 days)
gamma = 0.2     # Recovery rate (1/infectious period=5 days)
N = 1000

# Analytical R0 for both networks (configuration approximation):
k_er_mean = 8.036
k_er_2nd = 72.48
k_ba_mean = 7.968
k_ba_2nd = 138.024

# Q = (⟨k^2⟩-⟨k⟩)/⟨k⟩
q_er = (k_er_2nd - k_er_mean) / k_er_mean
q_ba = (k_ba_2nd - k_ba_mean) / k_ba_mean

# For SEIR, the effective R0 (average secondary infections) = beta/sigma * sigma/(gamma+mu) * q
# (Assume no vital dynamics, i.e., mu=0.)
# But often, for constant population:
#   R0 = beta * q / gamma
R0_er = beta * q_er / gamma
R0_ba = beta * q_ba / gamma

# Optionally, simulate deterministic SEIR ODE for homogeneous-mixing as reference
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def seir_ode(y, t, beta, sigma, gamma, N):
    S,E,I,R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Initial condition: 990 S, 5 E, 5 I, 0 R
y0 = [990, 5, 5, 0]
t = np.linspace(0, 200, 400)
sol = odeint(seir_ode, y0, t, args=(beta * k_er_mean, sigma, gamma, N))

# Save ODE solution plot
plt.figure(figsize=(7,5))
plt.plot(t, sol[:,0], label='S')
plt.plot(t, sol[:,1], label='E')
plt.plot(t, sol[:,2], label='I')
plt.plot(t, sol[:,3], label='R')
plt.xlabel('Time')
plt.ylabel('Number')
plt.title('Deterministic SEIR ODE (Homogeneous mixing)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'seir_ode_reference.png'))
plt.close()

results = {
    'R0_er': R0_er,
    'R0_ba': R0_ba,
    'q_er': q_er,
    'q_ba': q_ba
}
