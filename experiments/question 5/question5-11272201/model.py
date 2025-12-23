
# Defining model for the specified SIR epidemic on a configuration-model network
name = 'SIR_network'
compartments = ['S', 'I', 'R']
transitions = {
    'S -(I)-> I': 'T',   # Per-edge transmission probability T governs move from S to I via infectious contact (network-based)
    'I -> R': 'gamma'    # Standard recovery transition
}
reasoning_info = (
    'The model structure is a classic SIR compartmental system. No E or H compartments are needed per scenario; infection, '
    'recovery, and immunity dynamics are fully captured by S, I, and R. Mechanistic network epidemic transitions rely on edge-based transmission (parameterized by T), '
    'directly matching the percolation approach for configuration-model graphs necessary to analyze random and targeted immunization strategies. Vaccinated nodes are treated as removed (R) pre-epidemic, in line with standard SIR framing for sterilizing vaccines. '
    'This framework is the minimal yet sufficient structure for analyzing epidemic thresholds and intervention effects in the described network context. Parameters for transitions can be flexibly set post hoc to match analytic R0 and per-edge transmissibility derived for the experimental scenario.'
)