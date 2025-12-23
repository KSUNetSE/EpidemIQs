
# Model structure for Mutually Exclusive Competitive SIS on Multiplex Network
name = "SIS1SIS2-Multiplex-excl"
compartments = ["S", "I1", "I2"]
transitions = {
    "S -(I1_on_A)-> I1": "beta1",
    "S -(I2_on_B)-> I2": "beta2",
    "I1 -> S": "delta1",
    "I2 -> S": "delta2",
}
reasoning_info = '''
1. Compartment choice: Three mutually exclusive states per node (S, I1, I2) are necessary and sufficient for two exclusive pathogens where no co-infection is possible. This matches the biological requirement of mutual exclusion (superinfection impossible), which is critical for accurate competition representation.
2. Network multiplex structure: Each pathogen spreads only on its own network layer—modeled via separate adjacency matrices (A for Virus 1, B for Virus 2). This captures the mechanistic process whereby each transmission opportunity is limited to a subset of contacts, as in real-world respiratory viruses or malware.
3. Transition mapping: Infection transitions from S to I1 and S to I2 depend explicitly on counts of infected neighbors in the relevant layer—ensuring that the local network structure is faithfully used as the transmission pathway. Recovery transitions (I1->S, I2->S) are standard for SIS processes and reflect clearance and regained susceptibility.
4. Exclusion of I1<->I2 transitions ensures complete exclusivity: No node can be in dual infection, nor switch from one infection to the other without first becoming susceptible.
5. Parameterization (beta1, beta2, delta1, delta2) is standard and ensures generic adaptability, as supported in the cited literature (Sahneh & Scoglio 2013/2014, Wang et al. 2022, Gracy 2023). The structural rules directly map to the invasion threshold analysis central to the scenario goals.
6. The structure is fully compatible with mean-field theory, agent-based simulation, and ‘dynamic screening’ analysis for the invasion/coexistence/dominance regimes.
7. All modeling elements are maximally general, robust to alternative pathogen types/networks (as requested), and directly map to the quantitative criteria for prevalence and coexistence assessment. This compartmental structure is the minimal and sufficient basis for the required analyses.
'''
