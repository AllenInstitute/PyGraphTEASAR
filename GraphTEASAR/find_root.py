import numpy as np
from scipy import sparse


def find_graph_root(mesh_graph, valid=None):
    """
    Finds the root of a graph by finding the farthest point from the
    start_ind and then finding the farthest point from that point.

    Parameters
    ----------
    mesh_graph : scipy.sparse.csr_matrix
        sparse matrix of the graph
    valid : np.ndarray
        Binary mask of vertices to consider, by default None
        if None all vertices are considered

    Returns
    -------
    int
        best_root index of the chosen root, one side of the mutually farthest pair
    pred: np.ndarray
        predecessor array from the dijkstra algorithm for the best_root
    dist_to_root: np.ndarray
        distance array from the dijkstra algorithm for the best_root
    """
    if valid is None:
        valid = connected_component_slice(mesh_graph, return_boolean=True)
    vals = find_far_points_graph(mesh_graph, valid)
    best_root, _, pred, _, dist_to_root = vals
    return best_root, pred, dist_to_root, valid


def find_far_points_graph(mesh_graph, valid=None):
    """
    Finds the maximally far point along a graph by bouncing from farthest point
    to farthest point.

    Parameters
    ----------
    mesh_graph : scipy.sparse.csr_matrix
        sparse matrix of the graph
    valid : np.array
        boolean array of valid vertices to consider, by default None
        if None all vertices are considered

    Returns
    -------
    int
        best_root index of the farthest point
    int
        farthest index from the best_root
    pred
        predecessor array from the dijkstra algorithm for the best_root
    float
        distance from the best_root to the farthest point
    np.array
        distance array from the dijkstra algorithm for the best_root

    """
    max_observed_distance = 0
    current_distance = 0

    if valid is None:
        valid = np.ones(mesh_graph.shape[0], dtype=bool)
        current_root = 0
    else:
        current_root = np.where(valid)[0][0]

    best_root = current_root

    k = 0
    pred = None
    dist_to_root = None
    while 1:
        k += 1
        dist_to_current, predn = sparse.csgraph.dijkstra(
            mesh_graph, False, current_root, return_predecessors=True
        )
        dist_to_current[~valid] = -1
        far_from_current = np.argmax(dist_to_current)
        current_distance = dist_to_current[far_from_current]
        if current_distance > max_observed_distance:
            best_root = current_root
            current_root = far_from_current
            max_observed_distance = current_distance
            pred = predn
            dist_to_root = dist_to_current
        else:
            break

    return best_root, current_root, pred, max_observed_distance, dist_to_root


def connected_component_slice(G, ind=None, return_boolean=False):
    """
    Gets a numpy slice of the connected component corresponding to a
    given index. If no index is specified, the slice is of the largest
    connected component.

    Parameters
    ----------
    G : scipy.sparse.csr_matrix
        sparse matrix of the graph
    ind : int, optional
        index of the connected component, by default None
    return_boolean : bool, optional
        whether to return a boolean array of the connected component, by default False

    Returns
    -------
    np.ndarray
        array of vertex indices (or boolean array if return_boolean is True)
    """
    _, labels = sparse.csgraph.connected_components(G)
    if ind is None:
        label_vals, cnt = np.unique(labels, return_counts=True)
        ind = np.argmax(cnt)
        label = label_vals[ind]
    else:
        label = labels[ind]

    if return_boolean:
        return labels == label
    else:
        return np.flatnonzero(labels == label)
