
# Analytical explanation since sympy not available
# Return a formatted string with the analytical finding
analytical_result = '''
Analytical competitive SIS model on multiplex networks:
- Let τ1 = β1/δ1, τ2 = β2/δ2, λ1(A), λ1(B): largest eigenvalues of adjacency matrices.
- Each virus persists if τi > 1/λ1(Layer_i) (classic SIS threshold).
- If both τ1 > 1/λ1(A) and τ2 > 1/λ1(B), both can survive in principle.
- However, if the layers are identical (same adjacency/eigenvector structure), typically only one virus dominates (excludes the other).
- Coexistence is possible **only if** layers are sufficiently different in structure, especially if highly central nodes in layer A are distinct from those in layer B (such that dominance of one virus in its layer does not facilitate suppression in the other).

Key references:
- Faryad Darabi Sahneh, C. Scoglio (2014) PRE.
- Doshi, Mallick, Eun (2021) IEEE INFOCOM.
- Sahneh, Scoglio (2013) arXiv.
'''

analytical_result
