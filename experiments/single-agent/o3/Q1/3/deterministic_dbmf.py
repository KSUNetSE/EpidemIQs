
import numpy as np, scipy.integrate as integrate, os, networkx as nx, scipy.sparse as sparse, pandas as pd, json
output_dir=os.path.join(os.getcwd(),'output')
# Load small networks
G_er = sparse.load_npz(os.path.join(output_dir,'network_er_small.npz'))
G_ba = sparse.load_npz(os.path.join(output_dir,'network_ba_small.npz'))
# Get degree distribution
import collections
G_er_nx = nx.from_scipy_sparse_array(G_er)
G_ba_nx = nx.from_scipy_sparse_array(G_ba)

deg_counts_er = collections.Counter(dict(G_er_nx.degree()).values())
deg_counts_ba = collections.Counter(dict(G_ba_nx.degree()).values())
N_er = G_er_nx.number_of_nodes(); N_ba = G_ba_nx.number_of_nodes()
max_k = max(max(deg_counts_er), max(deg_counts_ba))
# Build pk arrays up to k_max
def pk_array(deg_counts,N):
    pk = np.zeros(max_k+1)
    for k,c in deg_counts.items():
        pk[k]=c/N
    return pk
pk_er = pk_array(deg_counts_er,N_er)
pk_ba = pk_array(deg_counts_ba,N_ba)
mean_er = sum(k*pk_er[k] for k in range(len(pk_er)))
mean_ba = sum(k*pk_ba[k] for k in range(len(pk_ba)))
# Model params
sigma=1/3
gamma=1/5
R0=2.5
# choose beta based on earlier formulas: beta = R0*gamma/q where q=(<k^2>-<k>)/<k>
second_er = sum((k**2)*pk_er[k] for k in range(len(pk_er)))
q_er=(second_er-mean_er)/mean_er
beta_er = R0*gamma/q_er
second_ba = sum((k**2)*pk_ba[k] for k in range(len(pk_ba)))
q_ba=(second_ba-mean_ba)/mean_ba
beta_ba = R0*gamma/q_ba
# Homogeneous param beta_h chosen earlier 0.5
beta_h = 0.5
# DBMF ODE system creator
def seir_dbmf_ode(t,y,pk,beta,mean_k):
    n_k = len(pk)
    S=y[0:n_k]; E=y[n_k:2*n_k]; I=y[2*n_k:3*n_k]; R=y[3*n_k:4*n_k]
    theta = sum(k*pk[k]*I[k] for k in range(n_k))/mean_k
    dS = -beta*np.arange(n_k)*S*theta
    dE = beta*np.arange(n_k)*S*theta - sigma*E
    dI = sigma*E - gamma*I
    dR = gamma*I
    return np.concatenate([dS,dE,dI,dR])

def integrate_dbmf(pk,beta,mean_k,days=160,dt=0.1):
    n_k=len(pk)
    # initial cond: small fraction infected across degrees relative to pk
    S0=pk.copy(); E0=np.zeros(n_k); I0=np.zeros(n_k); R0=np.zeros(n_k)
    # seed 0.0005 infected proportion uniformly
    seed=0.0005
    for k in range(n_k):
        remove = min(seed, S0[k])
        S0[k]-=remove
        I0[k]+=remove
    y0 = np.concatenate([S0,E0,I0,R0])
    sol = integrate.solve_ivp(seir_dbmf_ode,[0,days],y0,t_eval=np.arange(0,days,dt),args=(pk,beta,mean_k),rtol=1e-6,atol=1e-8)
    # aggregate
    I_total = sol.y[2*n_k:3*n_k,:].sum(axis=0)
    R_total = sol.y[3*n_k:4*n_k,:].sum(axis=0)
    return sol.t, I_total, R_total

t_er,I_er,R_er = integrate_dbmf(pk_er,beta_er,mean_er)
t_ba,I_ba,R_ba = integrate_dbmf(pk_ba,beta_ba,mean_ba)
# also homogeneous mixing ODE solved earlier
beta_h=0.5
sigma=1/3
gamma=1/5
E0=0.0005
I0=0.0005
S0=1-E0-I0
R0i=0

def seir_hom(t,y):
    S,E,I,R=y
    dS=-beta_h*S*I
    dE=beta_h*S*I - sigma*E
    dI=sigma*E - gamma*I
    dR=gamma*I
    return [dS,dE,dI,dR]
sol_h = integrate.solve_ivp(seir_hom,[0,160],[S0,E0,I0,R0i],t_eval=np.arange(0,160,0.1))
I_h=sol_h.y[2]
R_h=sol_h.y[3]
# compute metrics
metrics_det={}
for name,t_vec,I_vec,R_vec in [('Hom',sol_h.t,I_h,R_h),('ER',t_er,I_er,R_er),('BA',t_ba,I_ba,R_ba)]:
    peak_I=float(I_vec.max())
    peak_time=float(t_vec[I_vec.argmax()])
    final_size=float(R_vec[-1])
    metrics_det[name]={'peak_I':peak_I,'peak_time':peak_time,'final_size':final_size}
print(metrics_det)
# save curves to csv for plotting
for name,t_vec,I_vec,R_vec in [('Hom',sol_h.t,I_h,R_h),('ER',t_er,I_er,R_er),('BA',t_ba,I_ba,R_ba)]:
    df=pd.DataFrame({'time':t_vec,'I':I_vec,'R':R_vec})
    df.to_csv(os.path.join(output_dir,f'det_{name}.csv'),index=False)