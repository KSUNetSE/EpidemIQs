
# Prepare for final reporting: Analysis table of key results for appendix; summary metrics as LaTeX table.
import pandas as pd
import os

metrics = {
    'Peak Infected': 201,
    'Peak Time': 84.8,
    'Final Epidemic Size (R)': 769,
    'Epidemic Duration': 344.6,
    'Initial Susceptible': 990,
    'Population Size': 1000,
    'beta (network)': 0.01441,
    'gamma': 0.04,
    'Mean Degree': 6.95,
    'Second Degree Moment': 55.21,
    'q': 6.94
}

latex_table = """
\\begin{table}[ht!]
\\centering
\\caption{Summary of SIR Simulation Metrics on Erdös-Rényi Network (N=1000, $\\langle k \\rangle=7$)}
\\begin{tabular}{ll}
\\hline
Metric & Value \\
\\hline
Peak Infected & 201 \\
Peak Infection Time & 84.8 \\
Final Epidemic Size & 769 \\
Epidemic Duration & 344.6 \\
Initial Susceptible & 990 \\
Population Size & 1000 \\
Transmission Rate $\\beta$ & 0.01441 \\
Recovery Rate $\\gamma$ & 0.04 \\
Mean Degree $\\langle k\\rangle$ & 6.95 \\
Second Degree Moment $\\langle k^2 \\rangle$ & 55.21 \\
Mean Excess Degree $q$ & 6.94 \\
\\hline
\\end{tabular}
\\end{table}
"""

with open(os.path.join(os.getcwd(), "output", "summary_metrics_table.tex"), "w") as f:
    f.write(latex_table)

latex_table