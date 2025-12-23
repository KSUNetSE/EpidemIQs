
# Observed typo: 'ounput_curves' should be 'output_curves'. Fixing typo and running again; same logic, now will fix variable spelling for the time series container.
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

N = 1000
alpha = 0.1
m = 2
beta = 7.5
gamma = 1.0

max_steps = 365
nsim = 1000
final_sizes = []
time_to_peak = []
output_curves = [] # corrected typo here
fade_outs = 0

for sim in range(nsim):
    S = np.ones(N, dtype=int)
    I = np.zeros(N, dtype=int)
    R = np.zeros(N, dtype=int)

    patient_zero = np.random.choice(N)
    S[patient_zero] = 0
    I[patient_zero] = 1

    s_traj = [S.sum()]
    i_traj = [I.sum()]
    r_traj = [R.sum()]

    active = np.zeros(N, dtype=bool)

    for t in range(max_steps):
        # Recovery
        rec = np.where(I == 1)[0]
        if rec.size > 0:
            recovered = rec[np.random.rand(len(rec)) < gamma]
            I[recovered] = 0
            R[recovered] = 1
        # Activation
        active = np.random.rand(N) < alpha
        contacts = []
        for node in np.where(active)[0]:
            partners = np.random.choice([x for x in range(N) if x != node], m, replace=False)
            for partner in partners:
                contacts.append((node, partner))
        contacts += [(b, a) for (a, b) in contacts]
        # Infection on edges
        for a, b in contacts:
            if (I[a] and S[b]):
                if np.random.rand() < beta:
                    S[b] = 0
                    I[b] = 1
        s_traj.append(S.sum())
        i_traj.append(I.sum())
        r_traj.append(R.sum())
        if I.sum() == 0:
            break
    output_curves.append((s_traj, i_traj, r_traj))
    final_sizes.append(r_traj[-1])
    time_to_peak.append(np.argmax(i_traj))
    if r_traj[-1] < 0.05*N:
        fade_outs += 1

max_len = max(len(traj[0]) for traj in output_curves)
s_mat = np.zeros((nsim, max_len))
i_mat = np.zeros((nsim, max_len))
r_mat = np.zeros((nsim, max_len))
for k in range(nsim):
    l = len(output_curves[k][0])
    s_mat[k, :l] = output_curves[k][0]
    i_mat[k, :l] = output_curves[k][1]
    r_mat[k, :l] = output_curves[k][2]

time_vec = np.arange(max_len)
mean_S = s_mat.mean(axis=0)
mean_I = i_mat.mean(axis=0)
mean_R = r_mat.mean(axis=0)
lo_I = np.percentile(i_mat, 5, axis=0)
hi_I = np.percentile(i_mat, 95, axis=0)

output = pd.DataFrame({
    'time': time_vec,
    'mean_S': mean_S,
    'mean_I': mean_I,
    'mean_R': mean_R,
    'I_5pct': lo_I,
    'I_95pct': hi_I
})

os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)
result_csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
result_png_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
output.to_csv(result_csv_path, index=False)

plt.figure(figsize=(8,5))
plt.plot(time_vec, mean_S, label='S (mean)')
plt.plot(time_vec, mean_I, label='I (mean)')
plt.fill_between(time_vec, lo_I, hi_I, color='orange', alpha=0.2, label='I 5-95%')
plt.plot(time_vec, mean_R, label='R (mean)')
plt.xlabel('Timestep')
plt.ylabel('Nodes')
plt.title('SIR on activity-driven network (random patient zero; N=1000, nsim=1000)')
plt.legend()
plt.tight_layout()
plt.savefig(result_png_path)
plt.close()

# Chain-of-Thought for debugging temporal scenario:
# Reflection indicates that in both results-11 and results-12 (temporal cases), no epidemic spread is observed even for high beta, large m, and alpha.
# Suspected issue: beta=7.5 is a probability, but values > 1 cannot be interpreted as probability in np.random.rand() < beta, always true if beta > 1, yet code can misbehave due to incorrect input range.
# The requested setup is discrete-time, but the logic should be: beta is a probability [0,1]. Instead, we set beta = 1.0 (guaranteed transmission per S-I contact, as >1 means certain transmission under Bernoulli).
# Let's set beta=1.0 for temporal agent-based, and document correction. Also, ensure state update is isolated to avoid multiple infections in the same round. Will simulate only scenario 1, model 1 (results-11) first for correct epidemic propagation, then extend to model 2 if validated.

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

N = 1000
alpha = 0.1
m = 2
beta_raw = 7.5  # As in input, but needs adjustment to probability for Bernoulli draw
beta = min(1.0, beta_raw)  # Clamp to 1.0 max for probability
# Actually, all contacts are infectious (for this extreme testing, but normally beta should be in [0,1])
gamma = 1.0

max_steps = 365
nsim = 1000
final_sizes = []
time_to_peak = []
output_curves = []
fade_outs = 0

for sim in range(nsim):
    S = np.ones(N, dtype=int)
    I = np.zeros(N, dtype=int)
    R = np.zeros(N, dtype=int)
    patient_zero = np.random.choice(N)
    S[patient_zero] = 0
    I[patient_zero] = 1
    s_traj = [S.sum()]
    i_traj = [I.sum()]
    r_traj = [R.sum()]
    for t in range(max_steps):
        rec = np.where(I == 1)[0]
        if rec.size > 0:
            recovered = rec[np.random.rand(len(rec)) < gamma]
            I[recovered] = 0
            R[recovered] = 1
        active = np.random.rand(N) < alpha
        contacts = set()
        for node in np.where(active)[0]:
            partners = np.random.choice([x for x in range(N) if x != node], m, replace=False)
            for partner in partners:
                contacts.add((node, partner))
        contacts |= set((b, a) for (a, b) in contacts)
        # Apply all infection attempts to new_S and new_I to avoid overlapping updates
        new_S = S.copy()
        new_I = I.copy()
        for a, b in contacts:
            if I[a] == 1 and S[b] == 1:
                if np.random.rand() < beta:
                    new_S[b] = 0
                    new_I[b] = 1
        S = new_S
        I = new_I
        s_traj.append(S.sum())
        i_traj.append(I.sum())
        r_traj.append(R.sum())
        if I.sum() == 0:
            break
    output_curves.append((s_traj, i_traj, r_traj))
    final_sizes.append(r_traj[-1])
    time_to_peak.append(np.argmax(i_traj))
    if r_traj[-1] < 0.05*N:
        fade_outs += 1

max_len = max(len(traj[0]) for traj in output_curves)
s_mat = np.zeros((nsim, max_len))
i_mat = np.zeros((nsim, max_len))
r_mat = np.zeros((nsim, max_len))
for k in range(nsim):
    l = len(output_curves[k][0])
    s_mat[k, :l] = output_curves[k][0]
    i_mat[k, :l] = output_curves[k][1]
    r_mat[k, :l] = output_curves[k][2]

time_vec = np.arange(max_len)
mean_S = s_mat.mean(axis=0)
mean_I = i_mat.mean(axis=0)
mean_R = r_mat.mean(axis=0)
lo_I = np.percentile(i_mat, 5, axis=0)
hi_I = np.percentile(i_mat, 95, axis=0)

output = pd.DataFrame({
    'time': time_vec,
    'mean_S': mean_S,
    'mean_I': mean_I,
    'mean_R': mean_R,
    'I_5pct': lo_I,
    'I_95pct': hi_I
})

os.makedirs(os.path.join(os.getcwd(), 'output'), exist_ok=True)
result_csv_path = os.path.join(os.getcwd(), 'output', 'results-11.csv')
result_png_path = os.path.join(os.getcwd(), 'output', 'results-11.png')
output.to_csv(result_csv_path, index=False)

plt.figure(figsize=(8,5))
plt.plot(time_vec, mean_S, label='S (mean)')
plt.plot(time_vec, mean_I, label='I (mean)')
plt.fill_between(time_vec, lo_I, hi_I, color='orange', alpha=0.2, label='I 5-95%')
plt.plot(time_vec, mean_R, label='R (mean)')
plt.xlabel('Timestep')
plt.ylabel('Nodes')
plt.title('SIR on activity-driven network (random patient zero; revised, N=1000, nsim=1000)')
plt.legend()
plt.tight_layout()
plt.savefig(result_png_path)
plt.close()