
import pandas as pd, numpy as np, os, scipy.integrate as spi
current_dir=os.getcwd(); output_dir=os.path.join(current_dir,'output')
N=2000
beta=0.35714285714285715
sigma=1/3
gamma=1/7
init={'S':98,'E':1,'I':1,'R':0}
S0=init['S']/100*N; E0=init['E']/100*N; I0=init['I']/100*N; R0=0

def seir(y,t):
    S,E,I,R=y
    dS=-beta*S*I/N
    dE=beta*S*I/N - sigma*E
    dI=sigma*E - gamma*I
    dR=gamma*I
    return [dS,dE,dI,dR]

time=np.arange(0,181)
ret=spi.odeint(seir,[S0,E0,I0,R0],time)

df=pd.DataFrame(ret,columns=['S','E','I','R'])
df['time']=time
hm_csv=os.path.join(output_dir,'results-11.csv'); df.to_csv(hm_csv,index=False)
peak_I=int(df['I'].max()); peak_time=int(df['time'][df['I'].idxmax()]); final_size=int(df['R'].iloc[-1])
summary={'peak_I_hom':peak_I,'peak_time_hom':peak_time,'final_size_hom':final_size}
