
name = 'SIR'
compartments = ['S', 'I', 'R']
transitions = {'S -(I)-> I': 'beta', 'I -> R': 'gamma'}
reasoning_info = '''The compartmental mechanistic model suited to the scenario is the classic SIR model. The justification is as follows:

The disease course (as described) is acute and directly transmitted, with no explicit latent (E) or other clinical states; thus, only Susceptible, Infectious, and Recovered states are warranted. The dynamics are driven by (a) transmission via contact between an S and an I individual, and (b) recovery of an I individual. Network dynamics (activity-driven, memoryless, and either temporal or static weighted) only mediate who interacts and when—the mechanistic disease progression remains SIR.

Transitions are "S -> I", induced by contact with an infectious individual (mediated by an SI network edge at a given timestep—parametrized by infection probability beta), and "I -> R" (autonomous recovery, parametrized by gamma per time step). These cover all possible state transitions under the scenario definition.

This model design neither infers additional compartments (since none are mentioned or implied) nor assigns parameter values, focusing only on the logical and structural framework required for general scenario modeling across both temporal and aggregated network representations. This ensures maximal parsimony, fidelity to the process, and flexibility for comparative experimentation as described in the task.'''
