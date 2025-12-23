
# Fix for error above: correct inverse transform sampling for power-law. 'nu' was defined as np.random.rand(N) - OK, but pow_law_c variable originally had a sign error. We'll use the standard power-law sampling formula for exponent gamma on [amin, amax].
# For power-law P(a) ~ a^{-gamma}, inverse CDF method:
#   a = [(amax^{1-gamma} - amin^{1-gamma}) * r + amin^{1-gamma}]^{1/(1-gamma)}

import numpy as np
N = 10000
m = 1
T = 1000
activity_min = 0.005
activity_max = 1.0
gamma = 2.5
np.random.seed(43)

r = np.random.rand(N)
exp = 1 - gamma
amin_exp = activity_min ** exp
amax_exp = activity_max ** exp
activity = (r * (amax_exp - amin_exp) + amin_exp) ** (1 / exp)
activity = np.clip(activity, activity_min, activity_max)
activity[:5], activity.mean(), (activity**2).mean()