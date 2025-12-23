
# ----------- Summary Plots: Epidemic Final Size vs. Vaccination (Scenarios 1 & 4) ----------
import os
import pandas as pd
import matplotlib.pyplot as plt
# Scenario 1 - Poisson random
poiss1 = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-1attackrates.csv'))
# Scenario 4 - Tailored random
tailor4 = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-4attackrates.csv'))
plt.figure(figsize=(7,5))
plt.plot(100*poiss1.p_vac, 100*poiss1.final_attack, '-o', label="Poisson(3) random vac")
plt.plot(100*tailor4.p_vac, 100*tailor4.final_attack, '-s', label="Tailored 2/3/10 random vac")
plt.axvline(75, color='gray', ls='--', lw=1, label=r'Predicted $p_c$ (0.75)')
plt.xlabel("Vaccination fraction (%)")
plt.ylabel("Final epidemic size (% of population)")
plt.title("Final Epidemic Size vs. Vaccination Fraction (Random Vaccination)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-summaryattackrandvac.png'), dpi=150)
