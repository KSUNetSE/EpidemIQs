
# Model definition for the described scenario (SIRS-like on Watts-Strogatz network)

name = 'SIRS_Network_U-P-F'

compartments = ['U', 'P', 'F']

# Each transition is given in the requested formalism
the_transitions = {
    'U -(P)-> P': 'beta',   # Unaware node becomes Posting at rate beta per neighbor Posting (local, per edge)
    'P -> F': 'gamma',      # Posting becomes Fatigued at rate gamma (intrinsic, per node)
    'F -> U': 'xi'          # Fatigued becomes Unaware at rate xi (intrinsic, per node)
}

reasoning_info = '''
This model is constructed as follows:
1. Compartment Selection: Three compartments reflect all possible node states—Unaware (U), Posting/propagating (P), and Fatigued/immune (F)—directly matching SIRS-style information spreading with loss of immunity (forgetting). No compartments are left unrepresented by the scenario (no latent/incubating or additional, scenario does not admit more per context).
2. Transitions Intrinsic to Model Logic: 'U -(P)-> P' encodes classic per-contact spread (network-local infection, parameterized by beta, occurring per U–P adjacency), while 'P -> F' (fatigue, rate gamma) and 'F -> U' (forgetting, rate xi) are individual-level, memoryless Markov transitions as found in canonical SIRS-types and explicitly described in the scenario and supporting literature. 
3. Network-specific Formulation: The U→P endpoint is contact-driven and networked, explicitly implemented as per-neighbor interactions on a static Watts-Strogatz graph—consistent with both real social contagion and established mathematical models. Other transitions are standard node-level events.
4. Completeness and Sufficiency: Every plausible pathway demanded by scenario or by empirical/theoretical precedent is present, and no extraneous states are assumed, keeping model maximally parsimonious and interpretable.
5. Generalization and Analytical/Flexible Use: By parameterizing each rate abstractly (beta, gamma, xi) and not specifying values or initial conditions, model is well suited for both mean-field/analytical studies and full stochastic network simulations. The structure matches that used in academic literature to uncover oscillatory and steady-state regimes in social SIRS epidemics. 
6. Adherence to Task: This model only describes the mechanistic/parametric structure—not parameter values or initial composition—thereby following instructions precisely. 
'''

output = {'name': name, 'compartments': compartments, 'transitions': the_transitions, 'reasoning_info': reasoning_info}
