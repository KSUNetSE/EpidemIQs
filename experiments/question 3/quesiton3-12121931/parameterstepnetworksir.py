
# Parameters for both temporal and static SIR on networks
R0 = 3.0
mean_infectious_period = 1.0 # Disease: avg. infectious period = 1 time unit
gamma = 1.0 # Recovery rate (per unit time)

N = 1000 # Network size
mean_degree = 0.4 # ⟨k⟩ for both networks

# Calculate beta using bridge between well-mixed and network:
beta = (R0 * gamma) / mean_degree # Per S–I edge, per time unit

# Initial condition: 1 infected, rest susceptible, population size 1000
initial_infected = max(1, round(0.001 * N)) # At least one infected
initial_susceptible = N - initial_infected
initial_removed = 0

# Convert to percentages summing to 100 (rounded, integer, min 1% infection for simulation practicality)
I_percentage = max(1, round((initial_infected / N) * 100))
remaining = 100 - I_percentage
S_percentage = remaining
R_percentage = 0
initial_condition = {'S': S_percentage, 'I': I_percentage, 'R': R_percentage}

# Output values
beta, gamma, initial_condition