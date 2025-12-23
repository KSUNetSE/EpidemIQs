
import os, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out_dir = os.path.join(os.getcwd(), 'output')
paths = {'ER':'results-11.csv','BA':'results-12.csv'}
for name, csv in paths.items():
    df = pd.read_csv(os.path.join(out_dir, csv))
    plt.figure(figsize=(6,4))
    for col in ['S','E','I','R']:
        plt.plot(df['time'], df[col], label=col)
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title(f'SEIR dynamics on {name} network')
    plt.legend()
    fig_path = os.path.join(out_dir, f'figure_{name}.png')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
print('done')