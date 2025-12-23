
# Model specification for the scenario
name = "SIR (network, with sterilizing vaccination)"
compartments = ["S", "I", "R"]
transitions = {
    "S -(I)-> I": "beta (or T, per-contact infection rate/probability)",
    "I -> R": "gamma (recovery/removal rate)",
    "S -> R": "sterilizing vaccination (pre-epidemic intervention, not a dynamic rate)"
}
reasoning_info = '''
This model is built upon a SIR compartmental framework instantiated on a static (configuration model) contact network, as required by the scenario. The compartments S, I, R respectively represent susceptible, infectious, and removed (either recovered or vaccinated) individuals. The SIR dynamic suffices because there is no mention of a latent (E) phase in the scenario, and the disease is of SIR-type (no reinfection, direct transmission, eventual removal).

Transitions are:
- "S -(I)-> I": Susceptible node becomes infectious if it is connected to at least one infectious node. The transition is parameterized by the per-contact infection probability (T, or beta in classical models). On a network, this process is mediated by the adjacency structure, so infection only occurs along edges.
- "I -> R": Infectious nodes recover or are removed at a rate gamma.
- "S -> R": Vaccination with sterilizing immunity (pre-epidemic): selected nodes (either randomly or by degree) are removed from S (and the contact network) and placed in R. This transition is not dynamic (does not occur during the epidemic) but is an intervention implemented at t=0 to simulate vaccination campaigns. The details of who is vaccinated determine whether the vaccination is random or targeted; this is encoded in the set of nodes assigned to transition from S to R before the epidemic.

Vaccinated individuals are thus indistinguishable (in terms of epidemic risk) from recovered/immune individuals. The model structure is fully network-aware because it operates at the node/edge level (transmission is only possible along network connections). Random or targeted vaccination is solely realized by the initial S -> R assignments, not by a structural change in the SIR process.

This approach ensures accurate emulation of the scenario; the model is maximally flexible as required, and no unnecessary compartments are included. All differences in vaccination strategy are realized prior to t=0; the epidemic then proceeds via classical SIR transitions on the pruned network. This matches both accepted network epidemiology theory and standard practices in both analytical and simulation contexts.'''
