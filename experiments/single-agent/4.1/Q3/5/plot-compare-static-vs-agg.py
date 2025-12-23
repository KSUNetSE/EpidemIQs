
import matplotlib.pyplot as plt
import pandas as pd
import os

# Plotting again to ensure figure gets saved for reporting (for inclusion in the report!)
res1 = pd.read_csv(os.path.join('output','results-11.csv'))
resw = pd.read_csv(os.path.join('output','results-12.csv'))

plt.figure(figsize=(10,6))
plt.plot(res1['time'], res1['I']/1000, label='Unweighted-Static: I(t)')
plt.plot(resw['time'], resw['I']/1000, label='Weighted Aggregated: I(t)')
plt.plot(res1['time'], res1['R']/1000, linestyle='--', label='Unweighted-Static: R(t)')
plt.plot(resw['time'], resw['R']/1000, linestyle='--', label='Weighted Aggregated: R(t)')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Fraction of Population')
plt.title('SIR Epidemic: Comparison of Static vs Weighted Aggregated Time-Aggregate (Activity-driven)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('output','figure-compare-static-agg.png'))
plt.close()
