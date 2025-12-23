
# Analytical reasoning for SIS competition in a two-multiplex network
# Based on results from works such as: G. Wei et al., PhysRevE.89.062817 (2014);
# We perform the analysis:
# - Each virus propagates on one layer via SIS (exclusive infection constraint)
# - Beta1/delta1, Beta2/delta2 are the spreading and recovery rates respectively
# - The effective infection rate tau1 = beta1/(delta1 * lambda1(A)) and tau2 = beta2/(delta2 * lambda1(B)), where lambda1 are largest eigenvalues of adjacency of A,B
# - Both tau1, tau2 above single-virus threshold
# - Usually: coexistence is only possible in highly asymmetric attributed layers or for 'localized' eigenvectors
import numpy as np

# Let's formalize the mean-field equations and outline invariants
# Let x1[i] be prob node i infected by virus 1, x2[i] for virus 2 (exclusive x1[i] + x2[i] <= 1)
def competitive_sis_thresholds(A, B, beta1, delta1, beta2, delta2):
    # Largest eigenvalue as mean-field threshold denominator
    eig_A = np.linalg.eigvals(A)
    eig_B = np.linalg.eigvals(B)
    lambda1_A = np.max(np.real(eig_A))
    lambda1_B = np.max(np.real(eig_B))
    tau1 = beta1/(delta1*lambda1_A)
    tau2 = beta2/(delta2*lambda1_B)
    return {'tau1': tau1, 'threshold1': 1/lambda1_A, 'tau2': tau2, 'threshold2': 1/lambda1_B,
            'lambda1_A': lambda1_A, 'lambda1_B': lambda1_B}

# What determines coexistence vs. dominance?
def coexistence_criteria(A, B, beta1, delta1, beta2, delta2):
    # Under mean-field approximation, see references PhysRevE.89.062817
    # If layers are identical, the more aggressive virus globally prevails
    # Coexistence is possible if the principal eigenvectors of A and B (v_A and v_B) are localized/non-overlapping
    # In strongly overlapping networks, usually there is absolute dominance
    eigvals_A, eigvecs_A = np.linalg.eig(A)
    eigvals_B, eigvecs_B = np.linalg.eig(B)
    idx_A = np.argmax(np.real(eigvals_A))
    idx_B = np.argmax(np.real(eigvals_B))
    vA = np.real(eigvecs_A[:, idx_A])
    vB = np.real(eigvecs_B[:, idx_B])
    # Cosine similarity of dominant eigenvectors
    cosine_sim = np.dot(vA, vB)/(np.linalg.norm(vA)*np.linalg.norm(vB))
    # If similarity ~ 1: highly overlapping, no coexistence
    # If similarity low: can have coexistence if infection foci spatially separate
    return {'cosine_similarity': cosine_sim, 'localized_A': np.max(vA)/np.sum(vA), 'localized_B': np.max(vB)/np.sum(vB)}

# Generate two sample random networks for illustration
from networkx.generators.random_graphs import erdos_renyi_graph
import networkx as nx
N = 100
G_A = erdos_renyi_graph(N, 0.05, seed=1)
G_B = erdos_renyi_graph(N, 0.03, seed=2)
A = nx.to_numpy_array(G_A)
B = nx.to_numpy_array(G_B)

beta1, delta1 = 0.45, 0.18
beta2, delta2 = 0.32, 0.12

phase_diag = competitive_sis_thresholds(A, B, beta1, delta1, beta2, delta2)
coexist_diag = coexistence_criteria(A, B, beta1, delta1, beta2, delta2)

# Results summarize the analytical prediction
tauD1 = phase_diag["tau1"]>phase_diag["threshold1"]
tauD2 = phase_diag["tau2"]>phase_diag["threshold2"]
possible_coexist = coexist_diag['cosine_similarity'] < 0.92

print(f"tau1={phase_diag['tau1']}, tau1-threshold={phase_diag['threshold1']}\ntau2={phase_diag['tau2']}, tau2-threshold={phase_diag['threshold2']}")
print(f"Principal eigencosine similarity: {coexist_diag['cosine_similarity']}")
print(f"LocalizedA={coexist_diag['localized_A']} LocalizedB={coexist_diag['localized_B']}")
print(f"Coexistence expected: {possible_coexist and tauD1 and tauD2}")
print(f"Absolute dominance expected: {(not possible_coexist) and (tauD1 and tauD2)}")
# Return values
dict_result = {'tau1_val': phase_diag['tau1'], 'threshold1': phase_diag['threshold1'], 'tau2_val': phase_diag['tau2'], 'threshold2': phase_diag['threshold2'], 
    'cosine_similarity': coexist_diag['cosine_similarity'], 'localized_A': coexist_diag['localized_A'], 'localized_B': coexist_diag['localized_B'],
    'coexistence_expected': possible_coexist and tauD1 and tauD2, 'absolute_dominance_expected': (not possible_coexist) and (tauD1 and tauD2)}
