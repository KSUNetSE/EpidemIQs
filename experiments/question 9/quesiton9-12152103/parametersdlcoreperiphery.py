
# Parameter setting for SDL model on the specified core-periphery network with supplied rates and epidemic context
parameters = {'beta': 0.3, 'gamma': 0.5}

# Initial conditions specified: (a) seed in single high-degree core node, (b) seed in single random periphery node; all others Solvent
# Total banks: 100
def get_initial_conditions():
    # Case (a): single core seeding (core is 20% of 100 = 20 nodes)
    # 1 node distressed, 99 nodes solvent
    case_core = {'S': 99, 'D': 1, 'L': 0}
    # Case (b): single periphery seeding
    case_periph = {'S': 99, 'D': 1, 'L': 0}
    return [case_core, case_periph]

initial_conditions = get_initial_conditions()
# For reporting as percentages, convert values to integer % summing to 100 each, for population of 100 nodes
initial_conditions_percent = []
for cond in initial_conditions:
    total = sum(cond.values())
    pct_cond = {k: round((v / total) * 100) for k, v in cond.items()}
    # Correction to ensure sum to 100 due to rounding
    diff = 100 - sum(pct_cond.values())
    # Adjust S to absorb any rounding error
    pct_cond['S'] += diff
    initial_conditions_percent.append(pct_cond)

# Initial condition descriptions:
initial_condition_desc = [
    'Single high-degree core node initially Distressed (D=1 in core); all others Solvent (S)',
    'Single random periphery node initially Distressed (D=1 in periphery); all others Solvent (S)'
]

return_vars = ["parameters", "initial_condition_desc", "initial_conditions_percent"]
