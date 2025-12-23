
# Mechanistic SIR with pre-epidemic vaccination, for a network setting
model_name = 'SIRV-network'
compartments = ['S', 'I', 'R']
transitions = {
    'S -(I)-> I': 'beta',  # transmission along contact edge
    'I -> R': 'gamma',     # recovery/removal
    'S -> R': 'vaccination (pre-epidemic, prescribed fraction)'
}
reasoning_info = (
    'SIR is chosen because individuals can only be susceptible, infectious, or permanently removed (by recovery or vaccination). '
    'Vaccination is modeled as a pre-epidemic S -> R transition: nodes are removed before epidemic begins and become immune. '
    'All infection occurs over edges (network mediation), and R class includes all immunized/removed individuals. '
    'Transitions are: infection along edges at rate beta; recovery/removal at rate gamma; and S->R by pre-epidemic vaccination. '
    'This general structure supports random or targeted vaccination simply by selecting nodes before epidemic onset.'
)
