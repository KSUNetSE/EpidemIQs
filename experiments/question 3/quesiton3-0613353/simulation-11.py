
# simulation-11.py: Temporal Activity-Driven SIR, beta=0.60494, gamma=0.1, heterogeneity-adjusted
# Same as scenario 1, but beta lower

data_dir = os.path.join(os.getcwd(),'output')
activity_path = os.path.join(data_dir,'exp3-powerlaw-activity.npy')
edge_path = os.path.join(data_dir,'exp3-temporal-contact-edges.csv')
N = 10000
Tmax = 2000
n_sim = 120
beta = 0.60494  # Heterogeneity-aware
m = 2
gamma = 0.1

output_csv = os.path.join(os.getcwd(), 'output', 'results-11.csv')
output_png = os.path.join(os.getcwd(), 'output', 'results-11.png')

# Reuse the same function from scenario 1 (already defined above)
stats = run_temporal_activity_driven_sir(
    edge_path, activity_path, N, beta, gamma, m, Tmax, n_sim, output_csv, output_png
)