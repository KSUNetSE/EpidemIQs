
# Preparing a summary report string combining all extracted info in a concise readable form
summary_text = []

summary_text.append('Single-run SIR result files summary (units: fractions for proportions, counts for individuals, time as in file):\n')

for fp, metrics in summary_results.items():
    summary_text.append(
        f'{fp}:\n'
        f'  Total population (N_total): {metrics["N_total"]:.0f}\n'
        f'  Final epidemic size (R final count): {metrics["final_R"]:.1f} ({metrics["final_R_frac"]:.3f} fraction)\n'
        f'  Peak infection count (max I): {metrics["peak_I"]:.2f}\n'
        f'  Time of peak infection: {metrics["peak_time"]:.4f}\n'
        f'  Epidemic duration (first to last infected): {metrics["duration"]:.4f}\n'
        f'  Vaccination info: Not available\n'
    )

summary_text.append('\nSweep attack rate results:\n')
for fp, df in attackrate_summaries.items():
    summary_text.append(f'{fp}: vaccination fraction vs final attack rate')
    summary_text.append(df.to_string(index=False))
    summary_text.append('\n')

summary_report = '\n'.join(summary_text)