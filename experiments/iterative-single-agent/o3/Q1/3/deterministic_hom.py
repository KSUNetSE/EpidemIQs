
import numpy as np, scipy.integrate as integrate, os, json
beta_h=0.5 # ensures R0 2.5 with gamma=0.2
sigma=1/3
gamma=1/5
N=1.0
# initial
E0=0.00001; I0=0.00001; S0=N-E0-I0; R0i=0

def seir_ode(t,y):
    S,E,I,R = y
    dS=-beta_h*S*I
    dE=beta_h*S*I - sigma*E
    dI=sigma*E - gamma*I
    dR=gamma*I
    return [dS,dE,dI,dR]
sol=integrate.solve_ivp(seir_ode,[0,150],[S0,E0,I0,R0i],t_eval=np.arange(0,150,0.1))
I_peak=max(sol.y[2])
