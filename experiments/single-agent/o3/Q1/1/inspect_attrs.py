
import fastgemf as fg, scipy.sparse as sparse, os, numpy as np
out=os.path.join(os.getcwd(),'output')
G=sparse.load_npz(os.path.join(out,'network_ER.npz'))
SEIR_schema=(fg.ModelSchema('SEIR').define_compartment(['S','E','I','R']).add_network_layer('contact').add_edge_interaction(name='infect',from_state='S',to_state='E',inducer='I',network_layer='contact',rate='beta').add_node_transition(name='prog',from_state='E',to_state='I',rate='sigma').add_node_transition(name='rec',from_state='I',to_state='R',rate='gamma'))
conf=fg.ModelConfiguration(SEIR_schema).add_parameter(beta=.05,sigma=1/4,gamma=1/7).get_networks(contact=G)
init={'percentage':{'I':0.2,'S':99.8,'E':0,'R':0}}
sim=fg.Simulation(conf,initial_condition=init,stop_condition={'time':50},nsim=2)
print('before run attributes',dir(sim))
sim.run()
print('after',dir(sim))
print(type(sim.results))
