
model_A = {
    'name': 'SEIR',
    'compartments': ['S', 'E', 'I', 'R'],
    'transitions': {
        'S -(I)-> E': 'beta',
        'E -> I': 'sigma',
        'I -> R': 'gamma'
    },
}

model_B = {
    'name': 'Degree-structured SEIR',
    'compartments': ['S_k', 'E_k', 'I_k', 'R_k'],
    'transitions': {
        "S_k -(I_k')-> E_k": 'beta * k * Theta',
        'E_k -> I_k': 'sigma',
        'I_k -> R_k': 'gamma'
    },
}

reasoning_info = '''
Step-by-step chain-of-thought validation:

1. **State Choice:** The disease (modeled as flu-like) has a clear incubation (latency) period, and recovery is followed by immunity. This matches the minimal, mechanistically-required SEIR structure: Susceptible (S) → Exposed (E) → Infectious (I) → Recovered (R). No vital dynamics, no further states needed.

2. **Transitions:** The three transitions—(i) S to E (infection from I), (ii) E to I (latency progression), (iii) I to R (recovery)—are universally accepted for acute SEIR pathogens; parameterized by β, σ, γ respectively. No extra transitions are mechanistically relevant; disease-induced death or loss-of-immunity are excluded per problem context.

3. **Network Structure Effects:**
   - For homogeneous-mixing (Erdős–Rényi/random graph): All individuals are statistically identical and mean-field mass-action rates describe transitions. This justifies classic ODE model parametrization.
   - For degree-heterogeneous (scale-free) network: Degree stratification is compulsory since transmission risk is modulated by node’s connectivity; cluster model into S_k, E_k, I_k, R_k (per degree k). Transmission to S_k depends upon degree and infectiousness of neighbors, leading to "S_k -(I_k')-> E_k" with effective force proportional to contact degree (k) and infectious pressure (Θ).

4. **Parametric Exclusivity:** Only β, σ, γ are coded as transition parameters; no extra biological complexity or intervention is present. This matches all SEIR literature for non-vital, closed-population, flu-like epidemics.

5. **Comparability:** Both models use the same set of underlying compartments and transitions, only differing by the transmission mechanism as imposed by network structure. This ensures any differences in epidemic dynamics arise from network structure (the research goal), not arbitrary compartment expansion or overfitting.

6. **Minimality:** Adding compartments or transitions (such as pre-symptomatic, asymptomatic, etc.) is not justified without biological or empirical motivation. This preserves mechanistic clarity and facilitates direct comparison across the two network settings.

7. **Output:**
A. Homogeneous-mixing Model (SEIR):
   - Compartments: [S, E, I, R]
   - Transitions: 'S -(I)-> E', 'E -> I', 'I -> R' with parameters [β, σ, γ]
B. Degree-heterogeneous Model (Degree-structured SEIR):
   - Compartments: [S_k, E_k, I_k, R_k] stratified by degree k
   - Transitions: 'S_k -(I_k')-> E_k' (with force β·k·Θ), 'E_k -> I_k', 'I_k -> R_k'

**Conclusion:** This compartmental formulation fully and efficiently encodes the mechanistic, parametric, and topological features of the requested epidemic scenario. It is directly defensible by SEIR mathematical epidemiology, the structure of acute respiratory pathogens, and modern network modeling best-practices. The only distinction between the models is the network-driven transmission process; otherwise they are congruent at the epidemic-process level, per experimental design requirements.
'''
