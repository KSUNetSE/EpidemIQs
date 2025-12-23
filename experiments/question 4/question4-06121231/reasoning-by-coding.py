
#!python
import networkx as nx
import matplotlib.pyplot as plt

# The following code simply plots some conceptual pictures for the two layers structural features.
# For example, we illustrate a multiplex network with different edge overlap and degree correlations.

# We plot two networks.
# First network: high edge overlap, and second network: low edge overlap.

# Setup random network with 20 nodes for demonstration:
G = nx.erdos_renyi_graph(20, 0.2, seed=42)
H = nx.erdos_renyi_graph(20, 0.2, seed=43)  # different seed --> low overlap

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
pos = nx.spring_layout(G, seed=42)
nx.draw_networkx(G, pos, node_color='lightblue', edge_color='black')
plt.title('Layer A: High Edge Overlap (Conceptual)')

plt.subplot(1,2,2)
pos2 = nx.spring_layout(H, seed=43)
nx.draw_networkx(H, pos2, node_color='lightgreen', edge_color='black')
plt.title('Layer B: Low Edge Overlap (Conceptual)')

plt.show()

# Note: The above is an illustrative demonstration for conceptual visualization only.

# Here is the explanation:

explanation = '''
In competitive/exclusive SIS dynamics over a multiplex network with two layers, the structural features of each layer and their interplay play a pivotal role in the possibility of coexistence between competing infections.

1. Edge Overlap: 
   - High edge overlap means that many interactions occur on both layers, making the infection dynamics strongly coupled. In such cases, if one infection dominates a region, the high overlap means that the same individuals are repeatedly exposed, which can lead to competitive exclusion if one strain is more transmissible. However, it may also result in a robust shared set of transmission pathways that may somehow allow strain coexistence if both infections have similar parameter regimes and the competition is balanced spatially. 
   - Low edge overlap, on the other hand, decouples the layers. This leads to quasi-independent spreading patterns. If the layers are almost separate, one infection may dominate one layer while the other prevails in the second layer, fostering coexistence at the network level despite competition on a node-level (especially when nodes participate in both layers but with less correlated contact patterns).

2. Degree-Degree Correlation Across Layers:
   - If a node’s degree in layer A is highly correlated with its degree in layer B (i.e., hubs in one layer are also hubs in the other layer), those nodes become super-spreaders for both infections. This strong coupling might lead to a scenario where the infection with a slightly higher transmission benefit among hubs can outcompete the other across both layers. Thus, high interlayer degree correlation can inhibit coexistence by reinforcing competitive imbalances.
   - In contrast, if the degree correlations across layers are weak or negative, hubs in one layer might not necessarily be hubs in the other. This decoupling can provide niches where each infection can establish itself, thereby promoting coexistence by reducing direct competition in critical nodes.

3. Spectral Properties:
   - The spectral radius of the contact matrix (e.g., the adjacency matrix or its variants) in each layer determines the epidemic threshold. In the competitive SIS model, the threshold conditions for each infection depend on the leading eigenvalue and the corresponding eigenvector which localizes the infection on critical nodes.
   - When the principal eigenvalues of the two layers (or the effective matrices for each pathogen considering competition) are very similar, the competitive edge is minimized at the global level, which fosters conditions for coexistence. Conversely, if one layer exhibits a markedly larger spectral radius (indicating a lower epidemic threshold), that infection can quickly dominate and suppress the other infection.
   - Moreover, overlapping spectral features in a multiplex setting often imply that the system overall is balanced. However, if the eigen-directions (i.e., eigenvector centralities) are highly aligned, competition is reinforced on the most central nodes, whereas misaligned eigen-directions can lead to spatial or nodal segregation of infections and allow coexistence.

Analytical Takeaway:
The coexistence of competing exclusive infections in a multiplex network is thus favored by conditions where:
   • There is a moderate to low edge overlap allowing for partial decoupling of dynamics across layers.
   • Degree-degree correlation across layers is not too high, so that hubs (or critical nodes) are not the same for both layers, reducing direct competition on the most effective spreaders.
   • The spectral radii (and the associated eigenspaces) of the transmission matrices of the layers are closely matched but not perfectly aligned, thereby avoiding a scenario where one infection bénéficier de a strong majority of the high-centrality nodes.

In summary, structural decoupling (via reduced edge overlap and lower degree correlation) and balanced spectral properties are conducive to the stable coexistence of competitive exclusive infections. Conversely, high edge overlap, strong interlayer degree correlations, and significant discrepancies in the spectral properties typically promote competitive exclusion in favor of the more aggressive strain.
'''

print(explanation)
