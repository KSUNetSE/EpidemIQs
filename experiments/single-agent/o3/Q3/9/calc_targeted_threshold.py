
import numpy as np
x=0.1091
k0=(3-10*x)/(1-x)
k02=k0**2

def q_new(r):
    D=1 - x*r
    A = x*(1-r)*10 + (1-x)*k0
    B = x*(1-r)*100 + (1-x)*k02
    z=A/D
    k2=B/D
    return (k2 - z)/z
rs=np.linspace(0,1,101)
qs=[q_new(r) for r in rs]
# find threshold
thr=[r for r,q in zip(rs,qs) if q<1]
print(k0)
print(qs[:11])
print(thr[0] if thr else None)