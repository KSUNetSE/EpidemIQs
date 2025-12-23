
# Redefine output directory and filename
output_dir = os.path.join(os.getcwd(), 'output')
output_path = os.path.join(output_dir, 'epidemic_metrics_summary_corrected.png')

# Redo the plot with saved path
fig, ax = plt.subplots(figsize=(8, 5))

bar_labels = ['Time to 10% Infected (t_10)', 'Peak Prevalence (I_peak)']
bar_values = [time_to_10_percent, peak_prevalence]
bar_colors = ['blue', 'red']

bars = ax.bar(bar_labels, bar_values, color=bar_colors)

for i, v in enumerate(bar_values):
    ax.text(i, v + 0.01 * max(bar_values), f'{v:.2f}', ha='center', color='black')

ax.text(1, max(bar_values)*0.9, 'Time to 50% Infected (t_50): Not reached', fontsize=12, color='gray', ha='center')

y_max = max(bar_values) * 1.2
ax.set_ylim(0, y_max if y_max > 0 else 1)

ax.set_ylabel('Value (time units for t_10; proportion for I_peak)')
ax.set_title('Epidemic Metrics from SIR Simulation on ER Network')

plt.savefig(output_path)

output_path