
# Define the model structure - SEIBR, as justified
name = 'SEIBR'
compartments = ['S', 'E', 'I', 'B', 'R']
# Transitions: S-(I)->E (transmission), E->I (latent ends), I->B (symptoms/onset of bed-confinement), B->R (recovery)
transitions = {
    'S -(I)-> E': 'beta',
    'E -> I': 'sigma',
    'I -> B': 'gamma1',
    'B -> R': 'gamma2'
}
reasoning_info = (
    'The SEIBR model is chosen to mechanistically represent the transmission and observation process in the English boarding school influenza outbreak. '
    'Compartments reflect: Susceptible (S); Exposed/latent (E); infectious-but-not-yet-bed-confined (I, the sole infectious/transmitting state); Bed-confined (B, symptomatic but not transmitting); Recovered (R). '
    'Transitions track: (1) exposure from infectious contacts S-(I)->E, (2) latent-to-infectious E->I, (3) symptom onset/isolation I->B, (4) recovery B->R. '
    'Transmission is exclusively via I, reflecting the empirical insight that little/no spread occurs in bed-confined individuals. '
    'A stochastic block model (SBM) is specified for the contact structure, representing dormitory/class clustering, with no empirical network data but high within- vs low between-block connection probabilities. '
    'The model omits parameter values and initial seeding per instructions, focusing wholly on structural/parametric relationships. '
    'This structure is minimal/mechanistically justified for fitting both epidemic speed and observable B(t) curve in this classic outbreak scenario.'
)
