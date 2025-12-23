
# Save a table summary for main scenarios (i=1, j=4)
# Put scenario description versus output CSV/PNG result paths, and brief caption/metrics from before.
import os
import pandas as pd
summary = [
    {'scenario': 'Activity-driven temporal network SIR', 'results_csv': os.path.join(os.getcwd(), 'output', 'results-11.csv'), 'plot': os.path.join(os.getcwd(), 'output', 'results-11.png'), 'info':'Temporal, full causality, SIR with beta=6.0, gamma=1.0, N=1000, nsim=100'},
    {'scenario': 'Aggregated static network SIR',          'results_csv': os.path.join(os.getcwd(), 'output', 'results-12.csv'), 'plot': os.path.join(os.getcwd(), 'output', 'results-12.png'), 'info':'Aggregated ER, SIR with beta=0.00475, gamma=1.0, N=1000, nsim=100'},
    {'scenario': 'Analytical final size',                  'results_csv': os.path.join(os.getcwd(), 'output', 'results-13.csv'), 'plot': '', 'info':'Result for r = 1-exp(-R0*r), R0=3: ~94%'}
]
pd.DataFrame(summary).to_csv(os.path.join(os.getcwd(), 'output', 'results-14.csv'), index=False)
''