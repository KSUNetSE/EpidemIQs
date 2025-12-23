
# Directly create figure with main epidemic curves & annotate main points for the report
import matplotlib.pyplot as plt
import pandas as pd
import os

df = pd.read_csv(os.path.join(os.getcwd(), 'output', 'results-11.csv'))
plt.figure(figsize=(8,5))
plt.plot(df['time'], df['S'], label='Susceptible', color='blue')
plt.plot(df['time'], df['I'], label='Infected', color='red')
plt.plot(df['time'], df['R'], label='Recovered', color='green')

# annotate peak and final size
peak_I = df['I'].max()
peak_T = df['time'][df['I'].idxmax()]
final_R = df['R'].iloc[-1]
plt.scatter([peak_T], [peak_I], color='black')
plt.annotate(f'Peak I={int(peak_I)}', (peak_T, peak_I), xytext=(peak_T+5, peak_I+10), arrowprops=dict(arrowstyle='->'))
plt.scatter([df['time'].iloc[-1]], [final_R], color='purple')
plt.annotate(f'Final R={int(final_R)}', (df['time'].iloc[-1], final_R), xytext=(df['time'].iloc[-1]-20, final_R-50), arrowprops=dict(arrowstyle='->'))
# Duration
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('Epidemic Trajectories for SIR on ER Network (R0=2.5)')
plt.legend()
fig_epicurve = os.path.join(os.getcwd(),'output','epidemic_summary.png')
plt.savefig(fig_epicurve)
plt.close()
fig_epicurve
