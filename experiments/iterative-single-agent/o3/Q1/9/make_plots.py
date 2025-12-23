
import matplotlib.pyplot as plt, os, pandas as pd
output_dir = os.path.join(os.getcwd(),'output')

# load stochastic avg results
er = pd.read_csv(os.path.join(output_dir,'results-11.csv'))
ba = pd.read_csv(os.path.join(output_dir,'results-12.csv'))
# load deterministic 
er_d = pd.read_csv(os.path.join(output_dir,'results-21.csv'))
ba_d = pd.read_csv(os.path.join(output_dir,'results-22.csv'))

plt.figure(figsize=(6,4))
plt.plot(er['time'],er['I']/2000,label='ER stochastic')
plt.plot(er_d['time'],er_d['I']/2000,'--',label='ER deterministic')
plt.plot(ba['time'],ba['I']/2000,label='BA stochastic')
plt.plot(ba_d['time'],ba_d['I']/2000,'--',label='BA deterministic')
plt.xlabel('Time (days)')
plt.ylabel('Infectious fraction')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'figure1.png'))

# degree distributions histogram
import networkx as nx, numpy as np, scipy.sparse as sparse
G_er = nx.from_scipy_sparse_array(sparse.load_npz(os.path.join(output_dir,'er_network.npz')))
G_ba = nx.from_scipy_sparse_array(sparse.load_npz(os.path.join(output_dir,'ba_network.npz')))

deg_er = [d for _,d in G_er.degree()]
deg_ba = [d for _,d in G_ba.degree()]

plt.figure(figsize=(6,4))
plt.hist(deg_er,bins=range(max(deg_er)+1),alpha=0.7,label='ER',density=True)
plt.hist(deg_ba,bins=range(max(deg_ba)+1),alpha=0.7,label='BA',density=True)
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('P(k) (log scale)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'figure2.png'))

return_dict={'figs':['figure1.png','figure2.png']}
