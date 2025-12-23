
import numpy as np, scipy.sparse as sp, os, networkx as nx, pandas as pd, matplotlib.pyplot as plt
out_dir=os.path.join(os.getcwd(),'output')
# parameters
beta=0.03
sigma=1/5
gamma=1/7
T=180
nsim=30
initial_exposed_fraction=0.01

def run_simulation(adj_csr, tag):
    N=adj_csr.shape[0]
    adj=adj_csr.tocsr()
    results=[]
    for sim in range(nsim):
        S=np.ones(N,dtype=bool)
        E=np.zeros(N,dtype=bool)
        I=np.zeros(N,dtype=bool)
        R=np.zeros(N,dtype=bool)
        idx=np.random.choice(N,int(initial_exposed_fraction*N),replace=False)
        S[idx]=False; E[idx]=True
        daily_counts=[]
        for t in range(T+1):
            daily_counts.append([S.sum(),E.sum(),I.sum(),R.sum()])
            # infection process
            m=adj.transpose().dot(I.astype(int))  # number of infected neighbors per node (int array)
            p_infect=1-np.exp(-beta*m)
            rand=np.random.rand(N)
            new_E=(rand<p_infect) & S
            # progression
            rand=np.random.rand(N)
            new_I=(rand< (1-np.exp(-sigma))) & E
            rand=np.random.rand(N)
            new_R=(rand< (1-np.exp(-gamma))) & I
            # update
            S[new_E]=False; E[new_E]=True
            E[new_I]=False; I[new_I]=True
            I[new_R]=False; R[new_R]=True
        results.append(np.array(daily_counts))
    results=np.stack(results)
    mean_counts=results.mean(axis=0)
    times=np.arange(T+1)
    df=pd.DataFrame(mean_counts, columns=['S','E','I','R'])
    df.insert(0,'time',times)
    csv_path=os.path.join(out_dir,f'results-11{tag}.csv')
    df.to_csv(csv_path,index=False)
    plt.figure()
    for col in ['S','E','I','R']:
        plt.plot(times, df[col], label=col)
    plt.xlabel('Days');plt.ylabel('Individuals');plt.title(f'Mean SEIR dynamics {tag}')
    plt.legend()
    png_path=os.path.join(out_dir,f'results-11{tag}.png')
    plt.savefig(png_path)
    plt.close()
    return csv_path, png_path

adj_er=sp.load_npz(os.path.join(out_dir,'network_er.npz'))
adj_ba=sp.load_npz(os.path.join(out_dir,'network_ba.npz'))
paths={'er':run_simulation(adj_er,'1'),'ba':run_simulation(adj_ba,'2')}

# simulation-11.py
import numpy as np, os, pandas as pd, matplotlib.pyplot as plt, scipy.integrate as integrate
os.makedirs(os.path.join(os.getcwd(),'output'), exist_ok=True)
# parameters
N=5000
beta=0.35  # homogeneous mixing beta to get R0=beta/gamma=2.5 if gamma=0.14, but we use gamma=0.25; adjust
sigma=1/3  # 0.333
gamma=1/4  # 0.25
R0=beta/gamma
# Actually choose beta = R0*gamma =2.5*0.25=0.625
beta=0.625
# initial conditions
E0=50
I0=0
R0_pop=0
S0=N-E0-I0-R0_pop
# ODE system
def deriv(y,t,beta,sigma,gamma,N):
    S,E,I,R=y
    dSdt=-beta*S*I/N
    dEdt=beta*S*I/N - sigma*E
    dIdt=sigma*E - gamma*I
    dRdt=gamma*I
    return dSdt,dEdt,dIdt,dRdt

y0=(S0,E0,I0,R0_pop)

t=np.linspace(0,160,1601)

ret=integrate.odeint(deriv,y0,t,args=(beta,sigma,gamma,N))
S,E,I,R=ret.T
# save
output={'time':t,'S':S,'E':E,'I':I,'R':R}

pd.DataFrame(output).to_csv(os.path.join(os.getcwd(),'output','results-11.csv'),index=False)
plt.figure(figsize=(6,4))
plt.plot(t,I,label='Infected')
plt.plot(t,E,label='Exposed')
plt.plot(t,R,label='Recovered')
plt.xlabel('Time (days)');plt.ylabel('Population');plt.title('Deterministic SEIR (Homogeneous Mixing)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(),'output','results-11.png'))
