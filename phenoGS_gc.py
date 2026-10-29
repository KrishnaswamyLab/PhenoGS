import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

def reference_selection(data, n_clusters, random_state):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)
    return kmeans.cluster_centers_

def generate_gaussian(data, num_gauss_points, std=0.5):
    gaussian_points = []
    for point in data:
        points = np.random.normal(loc=point, scale = std, size =(num_gauss_points, len(point)))
        gaussian_points.append(points)
    return np.vstack(gaussian_points)


def build_affinity_matrix(O,N,sigma=1.0):
    dist_NO = pairwise_distances(N,O, squared= True)
    dist_ON = dist_NO.T
    K_NO = np.exp(-dist_NO/(2*sigma**2))
    K_ON = np.exp(-dist_ON/(2*sigma**2))
    K_NN = K_NO @ K_ON
    return K_NN

def graph_connectivity(O, N, K_NN, t):
    D = np.diag(np.sum(K_NN, axis=1))
    epsilon = 1e-8  # Small value to avoid division by zero
    D += np.eye(D.shape[0]) * epsilon
    D_inv = np.divide(1.0, D, out=np.zeros_like(D), where=(D != 0.0))
    P_NN = D_inv @ K_NN
    P_NN_powered = np.linalg.matrix_power(P_NN, t)
    N_graph_connectivity = P_NN_powered @ N
    all_points = np.vstack([O, N_graph_connectivity])
    return N_graph_connectivity, all_points


