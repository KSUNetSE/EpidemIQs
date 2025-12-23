
import matplotlib.pyplot as plt
import numpy as np

scenarios = list(results.keys())
fraction_vaccinated = [results[s]['fraction_vaccinated'] for s in scenarios]
final_size_frac = [results[s]['final_epidemic_size']['fraction'] for s in scenarios]
peak_inf_frac = [results[s]['peak_infection']['fraction'] for s in scenarios]
epidemic_durations = [results[s]['epidemic_duration']['mean'] for s in scenarios]

plt.figure(figsize=(12, 6))
plt.plot(fraction_vaccinated, final_size_frac, 'o-', label='Final Epidemic Size (fraction)')
plt.plot(fraction_vaccinated, peak_inf_frac, 's-', label='Peak Infection Prevalence (fraction)')
plt.xlabel('Fraction Vaccinated')
plt.ylabel('Fraction of Population (N=10000)')
plt.title('SIR Model Metrics vs Fraction Vaccinated (Different Scenarios)')
plt.legend()
plt.grid(True)
plt.savefig('output/sir_metrics_vs_vaccination.png')
plt.close()