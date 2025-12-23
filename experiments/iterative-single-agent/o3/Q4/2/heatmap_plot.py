
import pandas as pd, os, matplotlib.pyplot as plt, numpy as np

sweep=pd.read_csv(os.path.join(os.getcwd(),'output','sweep_results.csv'))
# pivot to heatmap values: difference final I1 - I2 positive -> 1 dom I1, negative -> -1 dom I2, 0 -> coexist
pivot=np.zeros((4,4))
for i,b1 in enumerate(sorted(sweep['beta1'].unique())):
    for j,b2 in enumerate(sorted(sweep['beta2'].unique())):
        row=sweep[(sweep['beta1']==b1)&(sweep['beta2']==b2)].iloc[0]
        if row['coexist']:
            val=0
        else:
            if row['I1_final']>row['I2_final']:
                val=1
            elif row['I2_final']>row['I1_final']:
                val=-1
            else:
                val=0
        pivot[i,j]=val

fig,ax=plt.subplots()
cmap=plt.cm.get_cmap('bwr',3)
im=ax.imshow(pivot, cmap=cmap, origin='lower', extent=[0.075,0.145,0.075,0.145])
ax.set_xlabel('beta2'); ax.set_ylabel('beta1'); ax.set_title('Dominance landscape: 1 meme1, -1 meme2, 0 coexist')
plt.colorbar(im, ticks=[-1,0,1])
fig_path=os.path.join(os.getcwd(),'output','heatmap_dom.png')
plt.savefig(fig_path, dpi=300)
