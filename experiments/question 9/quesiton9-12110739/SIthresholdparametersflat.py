
# Output for SI-threshold is meant as a parameter (threshold) list per network -- flat, one phi at a time.
# The signature expects Dict[str, List[float]] for parameters (so we only output φ, not full dicts per phi). 
# Supply only φ list for each network, and summarize E, B=φE in reasoning_info.

er_phi_values = [float(d['φ']) for d in parameter_scenarios['ER']]
ba_phi_values = [float(d['φ']) for d in parameter_scenarios['BA']]
parameters = {
    "ER": er_phi_values,
    "BA": ba_phi_values,
}
return_vars = ["parameters", "initial_conditions", "initial_condition_desc", "phi_c_ER", "phi_c_BA"]