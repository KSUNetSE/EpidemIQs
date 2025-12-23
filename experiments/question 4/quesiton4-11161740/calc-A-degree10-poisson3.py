
import math
lam=3
from math import exp, factorial
P=[exp(-lam)*lam**k/factorial(k) for k in range(21)]
P10=P[10]
# compute z=3
z=lam
# compute moment
A=sum(k*(k-1)*P[k] for k in range(21))-10*9*P10
A