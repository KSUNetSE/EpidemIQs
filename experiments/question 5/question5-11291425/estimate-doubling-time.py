
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Identify early phase: time up to the peak infection (less or equal to time_of_peak)
early_phase_data = data[data['time'] <= time_of_peak]
early_I_frac = early_phase_data['I'] / total_population

def find_early_exponential_phase(I_frac_series, min_fraction=0.001):
    """
    Find the data points corresponding to the early exponential phase when I/N > min_fraction
    """
    exp_phase_mask = I_frac_series > min_fraction
    return exp_phase_mask

exp_phase_mask = find_early_exponential_phase(early_I_frac)

# Only consider the time and log(I/N) where in exponential phase
time_exp = early_phase_data.loc[exp_phase_mask, 'time'].values.reshape(-1, 1)
log_I_exp = np.log(early_I_frac[exp_phase_mask].values).reshape(-1, 1)

# Linear regression fit
model = LinearRegression().fit(time_exp, log_I_exp)
growth_rate = model.coef_[0][0]  # slope

# Calculate doubling time: T_d = ln(2) / growth_rate
# growth_rate must be positive for doubling time to be valid
if growth_rate > 0:
    doubling_time = np.log(2) / growth_rate
else:
    doubling_time = np.nan

# Visualization of the early phase fit
plt.figure(figsize=(8,5))
plt.plot(time_exp, log_I_exp, 'o', label='log(I/N) data')
plt.plot(time_exp, model.predict(time_exp), 'r-', label='Linear fit')
plt.xlabel('Time')
plt.ylabel('log(Infected fraction I/N)')
plt.title('Early Phase Exponential Growth Fit')
plt.legend()
plt.tight_layout()
plt.savefig('output/early_phase_exponential_growth_fit.png')
plt.close()

doubling_time