
# PARAMETERS FOR COMPETITIVE SIS (EXCLUSIVE INFECTION) ON THE GIVEN MULTIPLEX NETWORK
# -- Derived from network spectral info and coexistence theory --

# Layer info:
lambda_A = 17.33  # spectral radius (largest eigenvalue), Layer A (BA)
lambda_B = 7.10   # spectral radius, Layer B (ER)
rho = -0.69       # eigenvector alignment (very negative = minimal overlap)

# Recovery rates (set to 1 per standard convention unless otherwise specified):
delta1 = 1.0
delta2 = 1.0

# Compute minimal thresholds for each infection [tau > 1/lambda]:
tau1_thresh = 1/lambda_A  # = 0.0577
	# => beta1 > 0.0577

tau2_thresh = 1/lambda_B  # = 0.1408
	# => beta2 > 0.1408

# Reasonable rate sets for phase exploration:
parameters_list = [
    # (β1, δ1, β2, δ2)
    {  # just above threshold (both viruses weak but viable)
        "beta1": 0.07, "delta1": 1.0, "beta2": 0.15, "delta2": 1.0
    },
    {  # well above threshold (robust co-circulation possible)
        "beta1": 0.14, "delta1": 1.0, "beta2": 0.25, "delta2": 1.0
    },
    {  # intermediate/balanced
        "beta1": 0.12, "delta1": 1.0, "beta2": 0.17, "delta2": 1.0
    }
]

# Initial conditions: 1% per infection type, rest susceptible
N = 1000
I1 = int(0.01 * N)
I2 = int(0.01 * N)
S = N - I1 - I2
initial_conditions = [
    {"S": S, "I1": I1, "I2": I2}
]
initial_condition_desc = [
    "Random 1% seeding per virus (distinct sets), others susceptible"
]

# Result for output (first parameter set as example, can loop for all in real scan)
parameters = parameters_list
