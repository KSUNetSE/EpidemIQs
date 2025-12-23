
import math, numpy as np
z=3
p=[math.exp(-z)*z**k/math.factorial(k) for k in range(0,60)]
S1=sum(k*p[k] for k in range(len(p)))
S2=sum(k*k*p[k] for k in range(len(p)))
q_orig=(S2-S1)/S1
print('q orig',q_orig)
P10=p[10]
S1
