
# The previous result for initial conditions was off due to rounding all 5 infected in N=1000:
# it should be S=995 (99.5%), I=5 (0.5%), R=0
init_S = 995
init_I = 5
init_R = 0

frac_S = round((init_S / 1000) * 100)
frac_I = round((init_I / 1000) * 100)
frac_R = round((init_R / 1000) * 100)
if (frac_S + frac_I + frac_R) != 100:
    frac_S = 100 - (frac_I + frac_R)
initial_conditions = [{"S": frac_S, "I": frac_I, "R": frac_R}]
initial_conditions