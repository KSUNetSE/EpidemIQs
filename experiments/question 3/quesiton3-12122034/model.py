
model = {
    "name": "SIR",
    "compartments": ["S", "I", "R"],
    "transitions": {
        "S -(I)-> I": "beta",
        "I -> R": "gamma"
    }
}
reasoning_info = '''
This SIR model is the archetypal mechanistic framework for directly transmitted, immunizing infectious diseases as specified in the scenario. It enumerates all necessary population states—Susceptible (S), Infected (I), and Recovered (R)—without extraneous complexity, in alignment with the scenario’s parametric and structural description. "S -(I)-> I" captures the infection of susceptibles through contact with infected individuals, where the (I) signifies the inducer state and the contact process is managed by the underlying network dynamics (activity-driven temporal or static aggregated). The "I -> R" transition models recovery at a specified rate, independent of network structure. By decoupling the abstract model from simulation-specifics (such as initial seeds and parameter values), the structure ensures applicability to studies aiming to explore how network representation (temporal versus static) impacts epidemic characteristics like final size and early outbreak probability. The chosen transitions fully specify the flow of individuals through all epidemic states in this context, and are analytically validated both by standard theory and scenario requirements. Any differences in outbreak metrics thus arise from network-level processes and the simulation context—not from overlooked states or mechanistic gaps.'''
