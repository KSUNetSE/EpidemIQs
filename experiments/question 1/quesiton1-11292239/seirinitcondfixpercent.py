
def fix_seir_init(popu, i, e):
    # Returns best integer percentage for init SEIR, ensuring 1% minimum for I when nonzero is desired
    n_inf, n_exp = i, e
    n_rec = 0
    n_sus = popu - n_inf - n_exp - n_rec
    # Calculate floats
    fra = [n_sus / popu, n_exp / popu, n_inf / popu, n_rec / popu]
    # Transform into % and round/adjust
    pix = [int(round(x*100)) for x in fra]
    diff = 100 - sum(pix)
    maxix = pix.index(max(pix))
    pix[maxix] += diff
    # Correction: ensure at least 1% for I, 0 if and only if requested/explained
    if i > 0 and pix[2] == 0:
        loss = 1 - pix[2]
        pix[2] = 1
        j = pix.index(max([pix[0], pix[1], pix[3]]))  # S/E/R
        pix[j] -= loss
    if e > 0 and pix[1] == 0:
        loss = 1 - pix[1]
        j = pix.index(max([pix[0], pix[2], pix[3]]))  # S/I/R
        pix[1] = 1
        pix[j] -= loss
    return {'S': pix[0], 'E': pix[1], 'I': pix[2], 'R': pix[3]}

init_rand = fix_seir_init(2000, 1, 3)
init_hubs = fix_seir_init(2000, 1, 3)

return_vars=["init_rand","init_hubs"]