
import pandas as pd, os, numpy as np, glob, json, math
output_dir = os.path.join(os.getcwd(),'output')
labels = ['baseline','random75','targeted_deg10_all','targeted_deg10_8']
metrics = {}
for idx,lab in enumerate(labels):
    df = pd.read_csv(os.path.join(output_dir, f'results-1{idx}.csv'))
    final_R = df['R'].iloc[-1]
    peak_I = df['I'].max()
    peak_time = df['time'][df['I'].idxmax()]
    metrics[lab] = {'final_R':int(final_R), 'peak_I':int(peak_I), 'peak_time':float(peak_time)}

metrics
import pandas as pd, os, numpy as np, json, glob
out_dir=os.path.join(os.getcwd(),'output')
summary_table=[]
for sim_id,label,vacc_frac in [('31','baseline',0.0),('32','random75',0.75),('33','degree10',None)]:
    df=pd.read_csv(os.path.join(out_dir,f'results-{sim_id}.csv'))
    record=df.iloc[0].to_dict()
    if vacc_frac is not None:
        record['vaccination']=vacc_frac
        record['attack_rate']=record['R_final']-vacc_frac
    else:
        # degree10 vaccination fraction previously computed
        record['vaccination']=0.0006
        record['attack_rate']=record['R_final']-0.0006
    record['scenario']=label
    summary_table.append(record)
summary_table

import pandas as pd, os, matplotlib
import matplotlib.pyplot as plt
import numpy as np
out_dir=os.path.join(os.getcwd(),'output')
labels=['baseline','random75','deg10']
metrics_list=[]
fig, ax=plt.subplots()
for idx,label in enumerate(labels):
    csv_path=os.path.join(out_dir, f'results-1{idx}.csv')
    df=pd.read_csv(csv_path)
    # compute metrics
    total_pop=df.iloc[0][['S','I','R']].sum()
    peak_I=df['I'].max()
    peak_time=df['I'].idxmax()
    final_R=df['R'].iloc[-1]
    final_size=final_R/total_pop
    epidemic_duration=df['time'].iloc[-1]
    metrics_list.append({'Scenario':label,'Peak_I':peak_I,'Peak_time':peak_time,'Final_R':final_R,'Final_size_frac':final_size,'Duration':epidemic_duration})
    ax.plot(df['time'], df['I']/total_pop, label=label)
ax.set_xlabel('Time steps')
ax.set_ylabel('Fraction Infected')
ax.legend()
fig_path=os.path.join(out_dir,'figure-1.png')
plt.savefig(fig_path)
plt.close()

# save metrics table
metrics_df=pd.DataFrame(metrics_list)
metrics_df.to_csv(os.path.join(out_dir,'metrics_summary.csv'), index=False)
metrics_df
