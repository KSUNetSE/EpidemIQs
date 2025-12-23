
import math
lam=3
p=[math.exp(-lam)*lam**k/math.factorial(k) for k in range(0,51)]
P10=p[10]
rem_prob=1-P10
mean_new=sum(k*p[k] for k in range(len(p)) if k!=10)/rem_prob
sec_new=sum(k*k*p[k] for k in range(len(p)) if k!=10)/rem_prob
q_new=(sec_new-mean_new)/mean_new
print(P10, mean_new, sec_new, q_new)