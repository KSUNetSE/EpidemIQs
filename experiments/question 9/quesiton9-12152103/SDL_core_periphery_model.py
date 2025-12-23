
# Model structure definition
name = 'SDL'
compartments = ['S', 'D', 'L']
transitions = {
    'S -(D)-> D': 'beta',  # Induced by stressed (Distressed) neighbor, network-based
    'D -> L': 'gamma'       # Spontaneous, node-based
}
reasoning_info = (
    "This SDL compartmental model is the most natural mechanistic choice for the scenario described: it captures the three core states "
    "(Solvent, Distressed, Liquidated) relevant to bank contagion. Transitions are defined both by network (contact-induced) and spontaneous "
    "processes. The S -> D transition is only possible through exposure to a directly connected Distressed node (network edge-mediated, beta-parameterized), "
    "while D -> L is a spontaneous, node-intrinsic (gamma-parameterized) transition. The network structure itself determines which nodes are at risk, but the "
    "model logic is agnostic to the values of beta/gamma, or the specific initial conditions or topological parameters, so it is flexible for analytic or simulation explorations. "
    "This abstraction aligns with both theoretical epidemic modeling (as in SIR-like network epidemics) and financial contagion studies, ensuring accurate mapping of state dynamics and capturing the influence of structural heterogeneity on epidemic speed and severity. "
    "The SDL model is, therefore, the standard, minimal, and maximally general parametric compartmental model for this network-based financial contagion application."
)
