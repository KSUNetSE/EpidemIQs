
import fastgemf as fg
import os

# --- Step 3 (revised): Define SIRS model schema (U-P-F cycle) and check model properties
model_schema = (
    fg.ModelSchema("SIRS-Upf")
    .define_compartment(['U', 'P', 'F'])
    .add_network_layer('contact_network')
    .add_edge_interaction(
        name='trend_infection',
        from_state='U',
        to_state='P',
        inducer='P',
        network_layer='contact_network',
        rate='beta'
    )
    .add_node_transition(
        name='fatigue',
        from_state='P',
        to_state='F',
        rate='gamma'
    )
    .add_node_transition(
        name='forgetting',
        from_state='F',
        to_state='U',
        rate='xi'
    )
)

# List compartments for verification
compartments = model_schema.compartments
# Try getting info on node and edge transitions (FastGEMF has .edge_transitions, .node_transitions)
edge_transitions = model_schema.edge_transitions
node_transitions = model_schema.node_transitions
compartments, edge_transitions, node_transitions
