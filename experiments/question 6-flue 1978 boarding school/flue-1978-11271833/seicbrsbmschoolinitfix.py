
# Correction to above: Initial 3 infected (I), so
# convert to percentage with at least 1% for I; rest susceptible - ensure at least 1 for I and sum is 100.
n = 763
I0 = 3
S0 = n - I0
E0=B0=C0=R0=0
init = [S0,E0,I0,B0,C0,R0]
perc = [int(round(i * 100 / n)) for i in init]
diff = 100 - sum(perc)
# To always ensure at least 1% infected:
if perc[2] == 0:
   perc[2] = 1
   perc[0] -= 1
if sum(perc) != 100:
    perc[0] += (100 - sum(perc))

init_perc = {'S': perc[0],'E': perc[1],'I': perc[2],'B': perc[3],'C': perc[4],'R': perc[5]}
init_perc