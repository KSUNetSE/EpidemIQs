
import numpy as np
import matplotlib.pyplot as plt

N = 1000  # population size

# Peak infection count and time to peak
I_peak = data['I'].max()
t_peak = data.loc[data['I'].idxmax(), 'time']

# Final number of susceptibles (last time point) and final epidemic size
S_inf = data['S'].iloc[-1]
final_epidemic_size = N - S_inf

# Epidemic duration: time when I first falls below 1 after peak
post_peak = data[data['time'] > t_peak]
end_condition = post_peak[post_peak['I'] < 1]
t_end = end_condition['time'].iloc[0] if not end_condition.empty else data['time'].iloc[-1]

# Doubling time estimation from early exponential growth phase
# Select early phase where I grows exponentially and I > 1
early_phase = data[(data['time'] <= t_peak) & (data['I'] > 1)]

# Log transform I for linear fit
log_I = np.log(early_phase['I'])
time_early = early_phase['time']

# Linear regression fit for log(I) = a + b*time
coeffs = np.polyfit(time_early, log_I, 1)
growth_rate = coeffs[0]  # b in exponential

doubling_time = np.log(2) / growth_rate if growth_rate > 0 else np.inf

# Theoretical final size from SIR equation (solve 1 - S_inf/N = exp(-R0*(1 - S_inf/N)))
from scipy.optimize import fsolve

beta = 0.3
ngamma = 0.1
R0 = beta / ngamma

final_size_func = lambda x: 1 - x - np.exp(-R0 * (1 - x))
solution = fsolve(final_size_func, 0.1)
theoretical_Sinf_ratio = solution[0]
theoretical_Sinf = theoretical_Sinf_ratio * N

# Plot S, I, R with peak infection and final size marked
plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['S'], label='Susceptible (S)')
plt.plot(data['time'], data['I'], label='Infected (I)')
plt.plot(data['time'], data['R'], label='Recovered (R)')
plt.axvline(t_peak, color='red', linestyle='--', label='Peak infection time')
plt.axhline(S_inf, color='green', linestyle='--', label='Final susceptible count')
plt.title('SIR Dynamics: Scenario 1 Well-mixed ODE SIR')
plt.xlabel('Time (days)')
plt.ylabel('Population count')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
import os
output_dir = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_path = os.path.join(output_dir, "SIR_Scenario1_summary.png")
plt.savefig(plot_path)
plt.close()

metrics = {
    'I_peak': I_peak,
    't_peak': t_peak,
    'S_inf': S_inf,
    'final_epidemic_size': final_epidemic_size,
    't_end': t_end,
    'doubling_time': doubling_time,
    'theoretical_Sinf': theoretical_Sinf
}

metrics