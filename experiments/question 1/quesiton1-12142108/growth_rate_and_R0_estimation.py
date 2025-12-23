
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define exponential growth function
exp_growth = lambda t, r, I0: I0 * np.exp(r * t)

# Subset first 10 days
first_10_days = data[data['time'] <= 10]

# Fit exponential growth on Infectious
params, cov = curve_fit(exp_growth, first_10_days['time'], first_10_days['I'], p0=[0.5, 1], maxfev=10000)

r = params[0]
I0_fit = params[1]
doubling_time = np.log(2)/r

# Theoretical R0 calculation for scale-free network
beta = 0.5
gamma_par = 0.143
mean_degree = 10

# Estimating <k^2>/<k> for gamma ~ 2.7-3.0
# For power law degree distribution with P(k) ~ k^-gamma, mean_degree = 10,
# rough approach: take <k^2>/<k> ~ 200 (approximate typical ratio for SF with exponent in range)
k_sq_div_k = 200
R0_theoretical = beta * k_sq_div_k / gamma_par

# Plot I(t) with peak
plt.figure(figsize=(10,6))
plt.plot(data['time'], data['I'], label='Infectious I(t)')
plt.scatter([peak_day], [max_I], color='red', label='Peak I')
plt.xlabel('Time (days)')
plt.ylabel('Number of Infectious individuals')
plt.title('Infectious population over time with peak')
plt.legend()
plt.grid(True)

plot_path = 'output/infectious_timeseries_with_peak.png'
plt.savefig(plot_path)

r, doubling_time, R0_theoretical, plot_path