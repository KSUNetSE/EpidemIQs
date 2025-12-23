
import math, os, json
def poisson_p(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)
P10 = poisson_p(10,3)
P10