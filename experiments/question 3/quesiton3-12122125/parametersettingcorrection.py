
# Correction: initial I from 1/1000 = 0.1%, which rounds to 0 in above code.
# To guarantee at least 1% initial infection for simulation feasibility,
# we set S=99, I=1, R=0 for percent-based IC (for a single-node seed in 1000, fraction is 0.001, but per instructions above, round up to at least 1%).
# For actual simulations, you can seed exactly 1 node, but in percentage output, we show 1% as minimal viable nonzero infection.

initial_condition_percent = [{'S': 99, 'I': 1, 'R': 0}]

return_vars = ['initial_condition_percent']
