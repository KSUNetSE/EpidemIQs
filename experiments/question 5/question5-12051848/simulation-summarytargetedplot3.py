
# Try removing $...$ around legend label, just write as regular text to avoid mathtext parse error
import os
import pandas as pd
import matplotlib.pyplot as plt
poiss1 = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-1attackrates.csv'))
tailor4 = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-4attackrates.csv'))
res2 = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-21.csv'))
res3 = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-31.csv'))
final_tail2 = res2['R'].iloc[-1] - res2['R'].iloc[0]
final_pois3 = res3['R'].iloc[-1] - res3['R'].iloc[0]
plt.figure(figsize=(7,5))
plt.plot(100*poiss1.p_vac, 100*poiss1.final_attack, '-o', alpha=0.5, label="Poisson(3) random vac (ref)")
plt.plot(100*tailor4.p_vac, 100*tailor4.final_attack, '-s', alpha=0.5, label="Tailored 2/3/10 random vac (ref)")
plt.scatter([11], [100*final_tail2/10000], color='red', marker='^', s=100, label="Tailored: all k=10 vaccinated (~11%)")
plt.scatter([0], [100*final_pois3/10000], color='blue', marker='v', s=100, label="Poisson: all k=10 vaccinated (~0.14%)")
plt.axvline(75, color='gray', ls='--', lw=1, label='Random vac pc ~ 75% (theory)')
plt.xlabel("Vaccination fraction (%) (horizontal axis; scenario specific)")
plt.ylabel("Final epidemic size (% of population)")
plt.title("Final Epidemic Size: Random vs Targeted Vaccination, Both Networks")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'results-summarytargeted.png'), dpi=150)
