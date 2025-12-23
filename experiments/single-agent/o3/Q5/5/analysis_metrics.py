
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
