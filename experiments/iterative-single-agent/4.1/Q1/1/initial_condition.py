
# Set up initial conditions for SIR model
# Population size
n = 1000
initial_infected = 10  # 1% infected
initial_recovered = 0
initial_susceptible = n - initial_infected - initial_recovered

initial_conditions = {'S': initial_susceptible, 'I': initial_infected, 'R': initial_recovered}
initial_conditions_frac = {k: int(100*v/n) for k,v in initial_conditions.items()}
initial_condition_type = ["randomly distributed"]
initial_conditions_frac, initial_condition_type