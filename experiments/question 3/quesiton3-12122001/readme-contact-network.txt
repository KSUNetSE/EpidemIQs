
Title: Contact Network Structures for SIR Temporal vs. Static Network Experiments

This directory contains population contact networks generated for a principled comparison of SIR dynamics on temporal (activity-driven) vs. static (ER-like aggregation) structures. See project plan and `construct-population-networks.py` for full pipeline.

1. `activitydriven-temporal-edgelist.csv`: Edge list for a synthetic activity-driven temporal network of N=1000 nodes for 5 steps (days). Each line is t,i,j for an ephemeral contact between nodes i and j on day t, generated under:
   - Each node independently activates per day with probability α=0.1
   - Upon activation, node forms m=2 undirected (random) ephemeral edges
   - No memory across days; edge list can be extended for full simulation duration (≥50 days) as needed

2. `ta-from-adt-aggr-adjacency.npz`: Adjacency matrix for time-aggregated static network (union of all contacts in 5-day window)
   - Also, `ta-from-adt-aggr-weighted-edgelist.csv` for counts of contact per edge across 5 days

3. `ta-er-statich-graph-adjacency.npz`, `ta-er-statich-graph-edgelist.csv`: Synthetic static ER random graph (N=1000, π≈0.002, ⟨k⟩≈2, Poisson(2) degree distribution). Represents time-aggregated contact network, for direct comparison against activity-driven aggregation. For reference.

4. Degree distribution plots: See *_degreedist.png and *_perday_mean_degree_hist.png for graphical comparison of typical degree statistics.

5. Centrality diagnostics (reported elsewhere in metadata):
   - Per-day mean degree and second moment (temporal model)
   - Aggregated mean degree and second moment
   - Same for ER static
   - Connected component and clustering diagnostics (for context; both networks are almost surely connected, and clustering is low as expected for random graphs)

Usage notes: For SIR model parameterization, see main project README/math_solution. Best practice is to use the calibrated β corresponding to network structure as described in the plan.

