
import pandas as pd, matplotlib.pyplot as plt, os
current_dir=os.getcwd(); output_dir=os.path.join(current_dir,'output')
for name in ['11','12','13']:
    df=pd.read_csv(os.path.join(output_dir,f'results-{name}.csv'))
    plt.figure()
    for comp in ['S','E','I','R']:
        plt.plot(df['time'],df[comp],label=comp)
    plt.xlabel('Days'); plt.ylabel('Individuals');
    title={'11':'Homogeneous-mixing SEIR','12':'Erdos-Renyi Network SEIR','13':'Scale-Free BA Network SEIR'}[name]
    plt.title(title); plt.legend()
    plt.savefig(os.path.join(output_dir,f'results-{name}.png'))
    plt.close()
