
import pandas as pd, matplotlib.pyplot as plt, os
output_dir=os.path.join(os.getcwd(),'output')
for name,csv,path_index in [('ER','results-21.csv',31),('BA','results-22.csv',32)]:
    df=pd.read_csv(os.path.join(output_dir,csv))
    plt.figure()
    for comp in ['S','E','I','R']:
        plt.plot(df['time'], df[comp], label=comp)
    plt.legend(); plt.xlabel('Time (days)'); plt.ylabel('Individuals'); plt.title(f'{name} network SEIR dynamics');
    save_path=os.path.join(output_dir,f'results-{path_index}.png')
    plt.savefig(save_path); plt.close()
print('done')