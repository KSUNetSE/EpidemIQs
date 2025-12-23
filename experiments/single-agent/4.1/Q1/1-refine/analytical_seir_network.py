
# Chain of Thought:
# Analytically, the deterministic mean-field SEIR for homogeneous mixing is standard ODEs. To incorporate degree-heterogeneous networks, we can use the degree-based mean-field approach: track S_k,E_k,I_k,R_k for each degree k.
# Let's construct both models, set basic parameters so both have the same basic R0, and produce time series for comparison.
# Calculate beta_hom for homogeneous: R0 = beta/gamma * (mean infectious duration)
# For networked, beta is related to network moments: beta_net = R0*gamma/q where q = (<k^2>-<k>)/<k> (see Pastor-Satorras, Vespignani etc.)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 1000
mean_k_er = 10.08
k2_er = 111.51
mean_k_ba = 9.95
k2_ba = 201.954

Tmax = 160
R0_fixed = 2.5
sigma = 1/3.0 # incubation rate (latent period = 3 days)
gamma = 1/6.0 # recovery rate (infectious period = 6 days)

# Homogeneous (well-mixed):
beta_hom = R0_fixed * gamma

def SEIR_hom(t, y):
    S, E, I, R = y
    dS = -beta_hom * S * I / N
    dE = beta_hom * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]

# Degree-based mean-field for BA network (Vespignani 2015): Only as an upper bound, not exact (makes degree bins)
kvals = np.unique(np.concatenate([np.arange(0,35,2),np.arange(0,18,1)]))
dist_ba,_ = np.histogram(np.round(np.linspace(0,38,N)), bins=np.append(kvals,100), density=True)
q_ba = (k2_ba - mean_k_ba) / mean_k_ba
beta_ba = R0_fixed * gamma / q_ba

# Population by degree bin
Nk = dist_ba * N
# Start: seed 10 infectives randomly
Ik = np.zeros_like(Nk)
Ik[0] = 10 # assign infected to lowest degree for simplicity (could distribute at random)
Sk = Nk - Ik
Ek = np.zeros_like(Nk)
Rk = np.zeros_like(Nk)

def SEIR_deg(t, y):
    Sk, Ek, Ik, Rk = y[:len(Nk)], y[len(Nk):len(Nk)*2], y[len(Nk)*2:len(Nk)*3], y[len(Nk)*3:]
    N_total = Sk+Ek+Ik+Rk
    theta = np.sum(np.arange(len(Sk))*Ik) / (N*np.sum(np.arange(len(Sk))*Nk)) # theta: prob a contact is with an I
    lambda_k = beta_ba * np.arange(len(Sk)) * theta
    dSk = -lambda_k * Sk
    dEk = lambda_k * Sk - sigma*Ek
    dIk = sigma*Ek - gamma*Ik
    dRk = gamma*Ik
    return np.concatenate([dSk, dEk, dIk, dRk])

# initial
y0_hom = [N-10, 0, 10, 0]
y0_deg = np.concatenate([Sk, Ek, Ik, Rk])

sol_hom = solve_ivp(SEIR_hom, [0,Tmax], y0_hom, dense_output=True, max_step=0.1)
cumI_hom = sol_hom.y[2]+sol_hom.y[3]
time_hom = sol_hom.t

sol_deg = solve_ivp(SEIR_deg, [0,Tmax], y0_deg, dense_output=True, max_step=0.1)
I_deg = np.sum(sol_deg.y[2*len(Nk):3*len(Nk)], axis=0)
cumI_deg = np.sum(sol_deg.y[2*len(Nk):, :], axis=0)
time_deg = sol_deg.t

plt.figure()
plt.plot(time_hom, cumI_hom, label='Cumulative I+R (Homogeneous)')
plt.plot(time_deg, cumI_deg, label='Cumulative I+R (Degree-Heterogeneous)')
plt.xlabel('Time (days)')
plt.ylabel('Cumulative Infections')
plt.title('Analytical SEIR: Homogeneous vs Degree-Heterogeneous')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('output', 'results-analytic.png'))
plt.close()

# Save results to CSVs
tab = np.stack([time_hom, cumI_hom], axis=1)
np.savetxt(os.path.join('output','results-analytic-hom.csv'), tab, delimiter=',', header='time,cum_I', comments='')
tab2 = np.stack([time_deg, cumI_deg], axis=1)
np.savetxt(os.path.join('output','results-analytic-het.csv'), tab2, delimiter=',', header='time,cum_I', comments='')
res = {'beta_hom': beta_hom, 'beta_ba': beta_ba, 'q_ba': q_ba}
res