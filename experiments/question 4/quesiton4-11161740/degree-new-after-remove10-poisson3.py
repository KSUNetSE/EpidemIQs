
import math, os
from math import exp, factorial
lam=3.0
P=[exp(-lam)*lam**k/factorial(k) for k in range(30)]
P10=P[10]
S=1-P10
Pprime=[P[k]/S if k!=10 else 0 for k in range(len(P))]

zprime=sum(k*Pprime[k] for k in range(len(Pprime)))
qprime=sum(k*(k-1)*Pprime[k] for k in range(len(Pprime)))/zprime
zprime, qprime
