
import numpy as np, math

f_vals = np.linspace(0.01,0.2,200)
solutions=[]
for f in f_vals:
    lam = (3-10*f)/(1-f)
    if lam<=0: continue
    second = (1-f)*(lam**2+lam)+100*f
    mean=3
    q=(second-mean)/mean
    if abs(q-4)<0.05:
        solutions.append((f,lam,q))

solutions[:5]
