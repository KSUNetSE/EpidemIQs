
P1=1/3
P3=4/7
P10=2/21
import numpy as np

def q_after(f):
    # fraction f of degree10 vaccinated
    num=6*P3 + 90*P10*(1-f)
    den=P1 + 3*P3 + 10*P10*(1-f)
    return num/den
for f in [0,0.5,1]:
    print(f,q_after(f))
# compute f required
f_req=( (12-3) )/(8.571428)  # earlier formula difference
print('f_req',f_req)