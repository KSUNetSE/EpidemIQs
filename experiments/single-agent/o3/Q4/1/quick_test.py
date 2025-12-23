
import os, numpy as np, pandas as pd, scipy.sparse as sparse, networkx as nx
from numpy.random import default_rng

output_dir=os.path.join(os.getcwd(),'output')
G_A=sparse.load_npz(os.path.join(output_dir,'layerA.npz'))
G_B2=sparse.load_npz(os.path.join(output_dir,'layerB2.npz'))

lambda1_A=np.linalg.eigvals(G_A.toarray()).real.max()
lambda1_B2=np.linalg.eigvals(G_B2.toarray()).real.max()

beta_factor=2.0
beta1=beta_factor/lambda1_A
beta2=beta_factor/lambda1_B2

delta1=1.0
delta2=1.0


def sim(beta1,beta2,dt=1.0):
    rng=default_rng(0)
    N=G_A.shape[0]
    neighbors_A=[G_A.indices[G_A.indptr[i]:G_A.indptr[i+1]] for i in range(N)]
    neighbors_B=[G_B2.indices[G_B2.indptr[i]:G_B2.indptr[i+1]] for i in range(N)]
    state=np.zeros(N,dtype=int)
    state[rng.choice(N,10,replace=False)]=1
    state[rng.choice(np.where(state==0)[0],10,replace=False)]=2
    time=[0]
    I1=[10]
    I2=[10]
    t=0
    while t<200:
        new_state=state.copy()
        for node in range(N):
            if state[node]==0:
                # infection attempts
                if any(state[nb]==1 for nb in neighbors_A[node]):
                    if rng.random()<1-np.exp(-beta1*dt):
                        new_state[node]=1
                        continue
                if any(state[nb]==2 for nb in neighbors_B[node]):
                    if rng.random()<1-np.exp(-beta2*dt):
                        new_state[node]=2
            elif state[node]==1:
                if rng.random()<1-np.exp(-delta1*dt):
                    new_state[node]=0
            elif state[node]==2:
                if rng.random()<1-np.exp(-delta2*dt):
                    new_state[node]=0
        state=new_state
        t+=dt
        time.append(t)
        I1.append((state==1).sum())
        I2.append((state==2).sum())
    return time,I1,I2

time,I1,I2=sim(beta1,beta2)
max(I1),max(I2)
