
# As sympy is not available, we will simply write out the equations as strings and their interpretation.
ODE_eqs = [
    "dS/dt = -beta * S * I / N",
    "dE/dt = beta * S * I / N - sigma * E",
    "dI/dt = sigma * E - gamma * I",
    "dR/dt = gamma * I"
]
# R0 for homogeneous mixing network
R0_hom = "beta / gamma"
# R0 for degree-heterogeneous network
R0_het = "(beta / gamma) * (<k^2> / <k>)"

ODE_eqs, R0_hom, R0_het