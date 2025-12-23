
# Additional chain-of-thought: Summarize the time-series for I (infected) in both scenarios for report (epidemic dies out or not, and statistics)
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('output/results-11.csv')
df2 = pd.read_csv('output/results-12.csv')

def summarize_epidemic(df):
    peak_I = df['I'].max()
    final_I = df['I'].iloc[-1]
    epidemic_happened = peak_I > 10  # arbitrary threshold
    epidemic_result = 'epidemic spreads and peaks' if epidemic_happened else 'no substantial epidemic, dies out quickly'
    t_peak = df['I'].idxmax()
    summary = {
        'peak_infection': int(peak_I),
        'final_infected': int(final_I),
        'epidemic_result': epidemic_result,
        'time_to_peak': int(t_peak)
    }
    return summary

summary1 = summarize_epidemic(df1)
summary2 = summarize_epidemic(df2)

# For appendix: save plot with both I time-series
plt.figure(figsize=(8,5))
plt.plot(df1['time'], df1['I'], label='Random vax (75%)')
plt.plot(df2['time'], df2['I'], label='Remove all deg 10')
plt.xlabel('Time')
plt.ylabel('Infected')
plt.legend()
plt.title('Infected over time: Random vs Targeted Vaccination')
plt.tight_layout()
plt.savefig('output/results-comparison.png')

summary_results = {'random_vax': summary1, 'deg10_vax': summary2}
summary_results
