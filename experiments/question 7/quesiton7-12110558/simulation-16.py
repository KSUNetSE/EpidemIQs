
# Revised Step: Comprehensive CSV gathering and plotting from all available simulation results
# This script ensures that all available kc scenario CSVs for both seedings are included in the sweep plot, and that barplots represent all kc values.
# 1. Find and load all results-ij.csv in output/, parse kc and seed type
# 2. Aggregate probability of global cascade for each (kc,seed_group)
# 3. Plot both sweep and barplot aggregating all available and missing cases
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

output_dir = os.path.join(os.getcwd(), 'output')
csvs = [f for f in os.listdir(output_dir) if re.match(r'results-\\d\\d\\.csv', f)]

# Map for kc values (by j, index) and scenario seeding (by i, 1=core, 2=periphery)
kc_label_map = {1:0.90, 2:0.80, 3:0.70, 4:1.00}  # but check for all
seed_label_map = {1:'core', 2:'periphery'}

kc_seed_prob = {}
for f in csvs:
    m = re.match(r'results-(\\d)(\\d)\\.csv', f)
    if m:
        i, j = int(m.group(1)), int(m.group(2))
        df = pd.read_csv(os.path.join(output_dir, f))
        kc = kc_label_map.get(j, None)
        seed = seed_label_map.get(i, None)
        if kc is not None and seed is not None:
            key = (kc, seed)
            prob = df['is_global_cascade'].mean() if 'is_global_cascade' in df else 0.0
            kc_seed_prob[key] = prob

# Collect all kc, two arrays for core and periphery
kc_vals = sorted(set([k for k,s in kc_seed_prob.keys()]))
kc_to_idx = {v:i for i,v in enumerate(kc_vals)}
y_core = [kc_seed_prob.get((kc,'core'), np.nan) for kc in kc_vals]
y_peri = [kc_seed_prob.get((kc,'periphery'), np.nan) for kc in kc_vals]

# Plotting: Sweep for all kc, 2 lines
plt.figure(figsize=(6,4))
plt.plot(kc_vals, y_core, 'o-r', label='Core seeded')
plt.plot(kc_vals, y_peri, 's-b', label='Periphery seeded')
plt.title('Global Cascade Probability vs. kc (theta=2)')
plt.xlabel('Core-Core Connection Probability (kc)')
plt.ylabel('Probability of Global Cascade (>50% failed)')
plt.ylim(0,1.05)
plt.xticks(kc_vals)
plt.legend()
for x,y in zip(kc_vals, y_core):
    if np.isnan(y): plt.annotate('NA', (x,0.1), color='red')
for x,y in zip(kc_vals, y_peri):
    if np.isnan(y): plt.annotate('NA', (x,0.2), color='blue')
plt.tight_layout()
plot_sweep = os.path.join(output_dir, 'results-12.png')
plt.savefig(plot_sweep)
plt.close()

# Barplot across all kc: Core vs Periphery for each kc
x = np.arange(len(kc_vals))
bar_width = 0.35
fig, ax = plt.subplots(figsize=(7,4))
rects1 = ax.bar(x - bar_width/2, y_core, bar_width, label='Core seeded', color='red')
rects2 = ax.bar(x + bar_width/2, y_peri, bar_width, label='Periphery seeded', color='blue')
ax.set_xlabel('Core-Core Connection Probability (kc)')
ax.set_ylabel('Probability of Global Cascade (>50% failed)')
ax.set_title('Global Cascade Probabilities by Seeding and kc')
ax.set_xticks(x)
ax.set_xticklabels([f"{v:.2f}" for v in kc_vals])
ax.set_ylim(0, 1.05)
ax.legend()
for i, y in enumerate(y_core):
    if np.isnan(y):
        ax.annotate('NA', (i - bar_width/2, 0.05), ha='center', color='k')
for i, y in enumerate(y_peri):
    if np.isnan(y):
        ax.annotate('NA', (i + bar_width/2, 0.15), ha='center', color='k')
fig.tight_layout()
plot_bar = os.path.join(output_dir, 'results-11.png')
plt.savefig(plot_bar)
plt.close()
_return = {'plot_sweep':plot_sweep, 'plot_bar':plot_bar,'kc_vals':kc_vals,'y_core':y_core,'y_peri':y_peri}
