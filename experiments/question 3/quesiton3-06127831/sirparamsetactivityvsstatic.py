
# Step 1: Define population and general parameters
N = 1000
R0 = 3.0
gamma = 1.0  # Recovery rate (per unit time, standard)

# Step 2: Network parameters
# (a) Activity-driven temporal network
m = 1  # number of edges created per activation
a = 3  # activation rate per node
r = 2 * m * a  # instantaneous contact rate per node (r=2ma=6)
# (b) Static aggregated network
mean_k = 181.26  # mean degree (from network description)

# Step 3: Calculate beta for both networks
# (a) Temporal (activity-driven)
beta_temporal = R0 * gamma / r  # (β/γ)·2ma = R0 => β = R0·γ/(2ma)
# (b) Static (aggregated)
beta_static = R0 * gamma / mean_k  # (β/γ)·⟨k⟩ = R0

# Step 4: Prepare initial conditions for both scenarios
# At t=0: 995 S, 5 I, 0 R (random assignment)
initial_conditions = [{'S': 995, 'I': 5, 'R': 0}]
initial_condition_type = ['random assignment (995 S, 5 I, 0 R among 1000 nodes)']

# Pack parameters for both scenarios
temporal_params = {'beta': beta_temporal, 'gamma': gamma}
static_params = {'beta': beta_static, 'gamma': gamma}
