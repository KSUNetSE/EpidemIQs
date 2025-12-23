
# Initial condition setup for N=10000
N = 10000
# Standard options: single seed OR 1% prevalence
init_single = {'S': (N-1)*100//N, 'I': 1*100//N, 'R': 0}
remainder = 100 - (init_single['S'] + init_single['I'] + init_single['R'])
if remainder != 0:
    init_single['S'] += remainder  # ensure integers sum to 100

init_1pct = {'S': int(100-1), 'I': 1, 'R': 0}

initial_condition_desc = [
    'Single infectious node (randomly selected), remainder susceptible; 0 recovered',
    '1% of nodes chosen at random infectious, 99% susceptible, 0% recovered'
]
initial_conditions = [init_single, init_1pct]

initial_condition_desc, initial_conditions