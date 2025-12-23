
import pandas as pd, matplotlib.pyplot as plt, os
output_dir=os.path.join(os.getcwd(),'output')
# Plot deterministic curves
plt.figure()
for name in ['Hom','ER','BA']:
    df=pd.read_csv(os.path.join(output_dir,f'det_{name}.csv'))
    plt.plot(df['time'],df['I'],label=name)
plt.xlabel('Time');plt.ylabel('Prevalence I');plt.legend();plt.title('Deterministic I over time');
plt.savefig(os.path.join(output_dir,'results-33.png'))
plt.close()
print('saved fig')