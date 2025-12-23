
# Reasoning and plan for model parameters and initial condition:
# 1. SIR model standard for community network epidemics
# 2. Use beta and gamma reflecting R0 = 2.5 (moderate outbreak)
#    mean_k = 23.688, mean_k2 = 581.06 => q = (<k^2> - <k>)/<k>
R0 = 2.5
gamma = 0.1     # recovery rate (1/gamma = 10 days infectious period)
q = (581.06 - 23.688)/23.688
beta = R0 * gamma / q
# Initial condition: 995 S, 5 I, 0 R (randomly distributed)
initial_conditions = [{'S': 995, 'I': 5, 'R': 0}]

# Output as model, parameters and IC info
model_name = 'SIR'
compartments = ['S', 'I', 'R']
transitions = {"S -(I)-> I": "beta", "I -> R": "gamma"}
parameters = {'beta': beta, 'gamma': gamma}
initial_condition_type = ["random"]
initial_condition_desc = ["5 randomly infected nodes, rest susceptible"]

# Save as Python dict for downstream agents
import json
with open(os.path.join(os.getcwd(), 'output', 'model_setup.json'), 'w') as f:
    json.dump({'model_name': model_name, 'compartments': compartments, 'transitions': transitions,
               'parameters': parameters, 'initial_conditions': initial_conditions,
               'initial_condition_type': initial_condition_type, 'initial_condition_desc': initial_condition_desc}, f)
# Print values for record
dict_out = {'model_name': model_name, 'compartments': compartments, 'transitions': transitions,
            'parameters': parameters, 'initial_conditions': initial_conditions,
            'initial_condition_type': initial_condition_type, 'initial_condition_desc': initial_condition_desc,
            'beta': beta, 'gamma': gamma, 'R0': R0, 'mean_k': float(mean_k), 'mean_k2': float(mean_k2), 'q': float(q)}
dict_out