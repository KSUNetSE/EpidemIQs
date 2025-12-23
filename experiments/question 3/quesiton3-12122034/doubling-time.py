
from scipy.optimize import curve_fit

# Define exponential growth function
exp_growth = lambda t, r, I0: I0 * np.exp(r * t)

# Select early phase where infected is non-zero and increasing
early_data = data[(data['SIR_I'] > 0) & (data['time'] < 0.1)].copy()

# Ensure infected is monotonically increasing in the selected phase
early_data = early_data[early_data['SIR_I'].diff().fillna(0) >= 0]

# Fit exponential growth model
popt, pcov = curve_fit(exp_growth, early_data['time'], early_data['SIR_I'], p0=[1.0, 1.0])

# Extract growth rate and calculate doubling time
growth_rate = popt[0]
doubling_time = np.log(2) / growth_rate if growth_rate > 0 else np.inf

# Return fitted parameters and doubling time
doubling_time, growth_rate