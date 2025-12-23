
import fastgemf as fg, scipy.sparse as sparse, os, numpy as np, pandas as pd
from pathlib import Path
current_dir=os.getcwd(); output_dir=os.path.join(current_dir,'output');

# parameters
N=2000
beta=0.35714285714285715
sigma=1/3
gamma=1/7
init_percent={'S':98,'E':1,'I':1,'R':0}

er_path=os.path.join(output_dir,'er_network.npz'); ba_path=os.path.join(output_dir,'ba_network.npz')
G_er=sparse.load_npz(er_path); G_ba=sparse.load_npz(ba_path)

SEIR_schema=(fg.ModelSchema('SEIR')
    .define_compartment(['S','E','I','R'])
    .add_network_layer('contact')
    .add_edge_interaction('infection',from_state='S',to_state='E',inducer='I',network_layer='contact',rate='beta')
    .add_node_transition('progression',from_state='E',to_state='I',rate='sigma')
    .add_node_transition('recovery',from_state='I',to_state='R',rate='gamma'))

cfg_er=(fg.ModelConfiguration(SEIR_schema)
         .add_parameter(beta=beta,sigma=sigma,gamma=gamma)
         .get_networks(contact=G_er))
cfg_ba=(fg.ModelConfiguration(SEIR_schema)
         .add_parameter(beta=beta,sigma=sigma,gamma=gamma)
         .get_networks(contact=G_ba))

ic={'percentage':init_percent}

sim_er=fg.Simulation(cfg_er,initial_condition=ic,stop_condition={'time':180},nsim=1)
sim_er.run()

sim_ba=fg.Simulation(cfg_ba,initial_condition=ic,stop_condition={'time':180},nsim=1)
sim_ba.run()


time_er,state_er,*_=sim_er.get_results()
time_ba,state_ba,*_=sim_ba.get_results()

import pandas as pd

er_df=pd.DataFrame({'time':time_er,'S':state_er[0], 'E':state_er[1],'I':state_er[2],'R':state_er[3]})
ba_df=pd.DataFrame({'time':time_ba,'S':state_ba[0], 'E':state_ba[1],'I':state_ba[2],'R':state_ba[3]})

# save
er_csv=os.path.join(output_dir,'results-12.csv'); ba_csv=os.path.join(output_dir,'results-13.csv')
er_df.to_csv(er_csv,index=False); ba_df.to_csv(ba_csv,index=False)

# compute metrics
peak_I_er=int(er_df['I'].max()); peak_time_er=int(er_df['time'][er_df['I'].idxmax()])
peak_I_ba=int(ba_df['I'].max()); peak_time_ba=int(ba_df['time'][ba_df['I'].idxmax()])
final_size_er=int(er_df['R'].iloc[-1]); final_size_ba=int(ba_df['R'].iloc[-1])

summary={'peak_I_er':peak_I_er,'peak_time_er':peak_time_er,'final_size_er':final_size_er,
         'peak_I_ba':peak_I_ba,'peak_time_ba':peak_time_ba,'final_size_ba':final_size_ba}
