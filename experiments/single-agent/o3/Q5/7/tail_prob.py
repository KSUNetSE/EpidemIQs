
import math, numpy as np
lam=3
p_tail=sum(math.exp(-lam)* lam**k / math.factorial(k) for k in range(10,50))
print(p_tail)
