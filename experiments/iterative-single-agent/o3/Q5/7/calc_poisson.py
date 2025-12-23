
import math
lam=3
p10=math.exp(-lam)*lam**10/math.factorial(10)
print('p10',p10)
# second moment for Poisson: var+mean^2 = lam + lam**2
m2=lam + lam**2
print('m2',m2)
