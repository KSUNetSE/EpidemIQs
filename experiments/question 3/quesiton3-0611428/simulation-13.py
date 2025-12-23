
# Scenario 3: Analytical Final Size Solution (i=1, j=3)
# Analytical result r = 1 - exp(-R0*r), here R0=3
# Numerically solve for r
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import os

R0 = 3.0
def final_size_eq(r):
    return r - (1 - np.exp(-R0 * r))

r0_guess = 0.9
r_sol, = fsolve(final_size_eq, r0_guess)

out = pd.DataFrame({'R0':[R0], 'final_size':[r_sol]})
out.to_csv(os.path.join(os.getcwd(), 'output', 'results-13.csv'), index=False)

# Save as text result for summary and return value
analysis_str = f'Analytical final size for R0=3: r = {r_sol:.4f} (fraction, i.e. {r_sol*100:.1f}%)'
analysis_str