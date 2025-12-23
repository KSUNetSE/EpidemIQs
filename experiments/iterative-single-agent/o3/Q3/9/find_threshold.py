
a=6/56
b=42/56
c=8/56

def q_r(r):
    denom=1 - a*r
    z=(2*b + 3*c + 10*a*(1-r)) / denom
    m2=(4*b + 9*c + 100*a*(1-r)) / denom
    return (m2 - z)/z
import numpy as np
rs=np.linspace(0,1,1001)
vals=[q_r(r) for r in rs]
idx=[i for i,v in enumerate(vals) if v<1]
threshold=rs[idx[0]] if idx else None
threshold
