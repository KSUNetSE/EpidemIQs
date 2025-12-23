
# Instead of percentage, let's report fraction and also test:
init_S = 995
init_I = 5
init_R = 0
N = 1000
frac_S = round((init_S / N)*100)
frac_I = round((init_I / N)*100)
frac_R = round((init_R / N)*100)
total = frac_S + frac_I + frac_R
result = {'frac_S': frac_S, 'frac_I': frac_I, 'frac_R': frac_R, 'total': total}
result
