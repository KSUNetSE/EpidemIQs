
import numpy as np
sol=None
for a in np.linspace(0,0.2,2001):
    lam=(3-10*a)/(1-a)
    if lam<=0: continue
    k2=100*a + (1-a)*(lam+lam**2)
    q=(k2-3)/3
    if abs(q-4)<1e-3:
        sol=a,lam,q
        break
sol