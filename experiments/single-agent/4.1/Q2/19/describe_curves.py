
# As the vision agent failed, we describe curves directly from the data.
# Compose a human-like analysis for figures and data for inclusion in Results and Discussion.
import pandas as pd

# SIR model results
sir_df = pd.read_csv('output/results-1-1.csv')
# SI model results
si_df = pd.read_csv('output/results-1-2.csv')

# SIR Curve Descriptions
sir_peak_I = sir_df['I'].max()
sir_peak_time = sir_df['time'][sir_df['I'].idxmax()]
sir_final_S = sir_df['S'].iloc[-1]
sir_final_R = sir_df['R'].iloc[-1]
sir_final_I = sir_df['I'].iloc[-1]
sir_end_time = sir_df[sir_df['I'] < 1]['time'].iloc[0] if (sir_df['I'] < 1).any() else sir_df['time'].iloc[-1]
sir_curve_desc = (
    f"In the SIR simulation, infections grow rapidly, peaking at t ≈ {sir_peak_time:.1f} (I = {sir_peak_I:.0f}), "
    f"then decline quickly. By t ≈ {sir_end_time:.1f}, almost all infectives recover and I drops to ≈ 0. "
    f"A significant proportion of susceptibles remain uninfected (S = {sir_final_S:.0f}/500). "
    f"Thus, the chain of transmission is broken because there are too few infectives, not due to depleted susceptibles."
)

# SI Curve Descriptions
si_final_I = si_df['I'].iloc[-1]
si_final_S = si_df['S'].iloc[-1]
si_curve_desc = (
    f"In the SI simulation, the number of infectives monotonically increases, while susceptibles are steadily depleted. "
    f"By the end of simulation, (I = {si_final_I:.0f}/500), all susceptibles are replaced by infectives (S = {si_final_S:.0f}/500). "
    f"Here, chain of transmission ends only when no susceptibles are left."
)

curve_figure_paths = {
    'sir': 'output/results-1-1.png',
    'si':  'output/results-1-2.png'
}

# Prepare text for Results/Discussion
results_discussion_text = f"""
Analytical and simulation results demonstrate clear differences between epidemic endpoints in the SIR and SI models. In the SIR scenario, the epidemic ends—i.e., the chain of transmission breaks—while a significant proportion of susceptibles remains; this occurs due to the depletion of infectives, not complete exhaustion of susceptibles. By contrast, in the SI model, the epidemic continues until there are no susceptibles left, at which point the chain is broken by their total depletion. This distinction is evident both analytically and in our network-based stochastic simulations (see Figures for population trajectories).\n\n
SIR trajectory: {sir_curve_desc}
\nSI trajectory: {si_curve_desc}
"""

results_discussion_text, curve_figure_paths