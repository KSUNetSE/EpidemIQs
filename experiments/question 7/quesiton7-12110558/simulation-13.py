
# --- CHAIN OF THOUGHT: Observation ---
# The plotting failed because missing kc values: not all networks/group files may exist, so in the loop the kc_plot_data arrays could end up shorter than the list of kc_vals.
# Action: Only plot for kc values where data for BOTH scenarios was collected. Assemble the list of actual kc's used based on collected data. 
# Step 2b: Fix by filtering kc and y-data for plotting together.

kc_vals = [0.7, 0.8, 0.9, 1.0]
kc_used = []
y_core = []
y_periphery = []
for idx in range(len(kc_vals)):
    # Only include if both y's (core+periphery) are present and length matches
    # Use original code's kc_plot_data but may not have data for all kc (files may be missing)
    try:
        # They should have been filled in matching order; skip if missing for either
        if len(kc_plot_data['core']) > idx and len(kc_plot_data['periphery']) > idx:
            y1 = kc_plot_data['core'][idx]
            y2 = kc_plot_data['periphery'][idx]
            kc_used.append(kc_vals[idx])
            y_core.append(y1)
            y_periphery.append(y2)
    except Exception as _:
        pass

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(kc_used, y_core, 'o-r', label='Core seeded')
plt.plot(kc_used, y_periphery, 's-b', label='Periphery seeded')
plt.title('Global Cascade Probability vs. kc (theta=2)')
plt.xlabel('Core-Core Connection Probability (kc)')
plt.ylabel('Probability of Global Cascade (>50% failed)')
plt.ylim(0,1.05)
plt.xticks(kc_used)
plt.legend()
plt.tight_layout()
plot_path = os.path.join(os.getcwd(), 'output', 'results-12.png')
plt.savefig(plot_path)
plt.close()
_return = {'plot_path': plot_path, 'kc_used': kc_used, 'y_core': y_core, 'y_periphery': y_periphery}
