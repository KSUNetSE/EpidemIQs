
# Mechanistic SIR model specification for a network-based scenario (activity-driven temporal and time-aggregated static)

model = {
    'name': 'Network-based SIR',
    'compartments': ['S', 'I', 'R'],
    # Notation: S -(I)-> I means S becomes I due to contact with I; parameter in quotes
    'transitions': {
        'S -(I)-> I': 'beta',      # Susceptible becomes Infectious upon contact with Infectious individual via present network edge (parameter: beta)
        'I -> R': 'gamma'          # Infectious recovers (and/or removed) at rate gamma
    }
}

reasoning_info = '''\
Chain-of-thought and justification:
- The scenario mandates a SIR compartmental model: S (susceptible), I (infectious), R (recovered/removed).
- No additional compartments (e.g., exposed, hospitalized, vaccinated) are specified, nor are there birth, death, importation, or waning immunity processes.
- The only two transitions permitted are (a) network-mediated infection (S-(I)->I, with per-edge infection parameter beta) and (b) recovery/removal (I->R, with independent per-node parameter gamma).
- For both the temporal and static network versions, infection only occurs if an S node has an I neighbor (dynamic or static edge, depending on network model). Recovery/removal is independent of the network, following a Poisson process with rate gamma per infectious node.
- Transition structure is modular and parameterized, suitable for use with both time-varying activity-driven networks (edges update each step) and static networks (edges fixed over epidemic timescale), allowing for downstream simulation or analytic derivation.
- This structure is the gold standard and is universally used for directly-transmitted, acute immunizing diseases, as in the present scenario. All steps in defining the model and transitions were validated with mathematical/epidemiological theory and the explicit scenario description (see cited Chain of Thought Plan).
- By abstracting at this level (without parameter values), the model is maximally general and can be instantiated for any relevant parameter set or initial condition; it fits perfectly into simulation or analytic workflows for both network structures described.'''

model, reasoning_info