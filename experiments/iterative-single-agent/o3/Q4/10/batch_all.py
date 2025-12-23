
import os, numpy as np, scipy.sparse as sparse, fastgemf as fg, pandas as pd, random
from tqdm import trange

# function to run multiple sims and output metrics

def run_batch(A_path,B_path,layerB_name,suffix,beta_factor=1.0,nsims=10):
    A=sparse.load_npz(A_path)
    B=sparse.load_npz(B_path)
    from scipy.sparse import linalg as spla
    lambdaA=float(np.real(spla.eigs(A,k=1,which='LR',return_eigenvectors=False)[0]))
    lambdaB=float(np.real(spla.eigs(B,k=1,which='LR',return_eigenvectors=False)[0]))
    beta1=beta_factor*1.5/lambdaA
    beta2=beta_factor*1.5/lambdaB
    delta1=delta2=1.0
    model_schema=(fg.ModelSchema('SI1I2S')
        .define_compartment(['S','I1','I2'])
        .add_network_layer('layerA')
        .add_network_layer(layerB_name)
        .add_edge_interaction('inf1',from_state='S',to_state='I1',inducer='I1',network_layer='layerA',rate='beta1')
        .add_edge_interaction('inf2',from_state='S',to_state='I2',inducer='I2',network_layer=layerB_name,rate='beta2')
        .add_node_transition('rec1',from_state='I1',to_state='S',rate='delta1')
        .add_node_transition('rec2',from_state='I2',to_state='S',rate='delta2')
    )
    model_config=(fg.ModelConfiguration(model_schema)
        .add_parameter(beta1=beta1,beta2=beta2,delta1=delta1,delta2=delta2)
        .get_networks(layerA=A,**{layerB_name:B})
    )
    n=A.shape[0]
    summary=[]
    for k in range(nsims):
        I1_nodes=np.random.choice(n,int(0.05*n),False)
        remain=list(set(range(n))-set(I1_nodes))
        I2_nodes=np.random.choice(remain,int(0.05*n),False)
        X0=np.zeros(n,int); X0[I1_nodes]=1; X0[I2_nodes]=2
        init={'exact':X0}
        sim=fg.Simulation(model_config,initial_condition=init,stop_condition={'time':300},nsim=1)
        sim.run()
        t, counts,*_=sim.get_results()
        summary.append(counts[:,-1])
    summary=np.array(summary)
    mean_final=summary.mean(axis=0)
    coexist=(mean_final[1]>0.05*n) and (mean_final[2]>0.05*n)
    out={'beta_factor':beta_factor,'mean_S':mean_final[0],'mean_I1':mean_final[1],'mean_I2':mean_final[2],'coexist':coexist}
    # save summary csv
    df=pd.DataFrame(summary,columns=['S','I1','I2'])
    df.to_csv(os.path.join(os.getcwd(),'output',f'results-{suffix}.csv'),index=False)
    return out

A_path=os.path.join(os.getcwd(),'output','layer_A.npz')
outs=[]
outs.append(run_batch(A_path,os.path.join(os.getcwd(),'output','layer_B.npz'),'layerB','21'))
outs.append(run_batch(A_path,os.path.join(os.getcwd(),'output','layer_B2.npz'),'layerB2','22'))
outs.append(run_batch(A_path,os.path.join(os.getcwd(),'output','layer_B3.npz'),'layerB3','23'))
outs.append(run_batch(A_path,os.path.join(os.getcwd(),'output','layer_B4.npz'),'layerB4','24'))
outs