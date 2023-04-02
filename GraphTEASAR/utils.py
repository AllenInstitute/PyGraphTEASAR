from scipy import sparse
import numpy as np


def create_spatial_skeleton(all_paths, vertices):
    """Creates a spatial skeleton (vertices and edges)
      from a list of paths and a set of vertices

    Args:
        all_paths (list): A list of paths, where each path is a list of vertex indices
        vertices (np.array): A MxK numpy array of vertex locations

    Returns:
        np.array: A NxK numpy array of vertex locations
          to only includes vertices referenced in all_paths
        np.array: A Mx2 numpy array of edges
            where each edge is a pair of vertex indices in the first array
        np.array: A Mx2 numpy array of the original vertex indices
            these are indices into the original vertices array
    """
    all_edges = all_paths_to_edges(all_paths)
    new_verts, new_edges, orig_indices = reduce_verts(vertices, all_edges)
    return new_verts, new_edges, orig_indices


def reduce_verts(verts, faces):
    """removes unused vertices from a graph or mesh

    Parameters
    ----------
    verts : numpy.array
        NxD numpy array of vertex locations
    faces : numpy.array
        MxK numpy array of connected shapes (i.e. edges or tris)
        (entries are indices into verts)

    Returns
    -------
    np.array
        new_verts, a filtered set of vertices
    np.array
        new_face, a reindexed set of faces (or edges)
    np.array
        used_verts, the index of the new_verts in the old verts

    """
    used_verts = np.unique(faces.ravel())
    new_verts = verts[used_verts, :]
    new_face = np.zeros(faces.shape, dtype=faces.dtype)
    for i in range(faces.shape[1]):
        new_face[:, i] = np.searchsorted(used_verts, faces[:, i])
    return new_verts, new_face, used_verts


def all_paths_to_edges(all_paths):
    """convert a list of paths to a list of edges

    Parameters
    ----------
    all_paths : list of lists of vertex indices

    Returns
    -------
    list of lists of edges
    """
    all_edges = []
    for comp_paths in all_paths:
        all_edges.append(paths_to_edges(comp_paths))
    if len(all_edges) > 0:
        tot_edges = np.vstack(all_edges)
    else:
        raise ValueError("No edges found!")
    return tot_edges


def paths_to_edges(path_list):
    """
    Turn an ordered path into an edge list.

    Parameters
    ----------
    path_list : list
        list of paths, where each path is a list of vertex indices

    Returns
    -------
    np.ndarray
        Mx2 array of vertex indices corresponding to edges, dtype should be int.
    """
    arrays = []
    for path in path_list:
        p = np.array(path)
        e = np.vstack((p[0:-1], p[1:])).T
        arrays.append(e)
    return np.vstack(arrays)


def create_spatial_csgraph(vertices, edges, euclidean_weight=True, directed=False):
    """
    Builds a csr graph from vertices and edges, with optional control
    over weights as boolean or based on Euclidean distance.

    Parameters
    ----------
    vertices : np.ndarray
        NxK array of vertex coordinates in a K-dimensional space.
    edges : np.ndarray
        Mx2 array of vertex indices corresponding to edges, dtype should be int.
    euclidean_weight : bool, optional
        whether to use Euclidean distance as edge weights, by default True
    directed : bool, optional
        whether to create a directed graph, by default False

    Returns
    -------
    scipy.sparse.csr_matrix
        sparse matrix of the graph
    """
    if vertices.ndim != 2 or vertices.shape[1] < 1:
        raise ValueError("vertices must be a NxK array with K > 0")

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must be a Mx2 array")

    edges = edges[edges[:, 0] != edges[:, 1]]
    if euclidean_weight:
        xs = vertices[edges[:, 0]]
        ys = vertices[edges[:, 1]]
        weights = np.linalg.norm(xs - ys, axis=1)
        use_dtype = np.float32
    else:
        weights = np.ones((len(edges),)).astype(np.int8)
        use_dtype = np.int8

    if directed:
        edges = edges.T
    else:
        edges = np.concatenate([edges.T, edges.T[[1, 0]]], axis=1)
        weights = np.concatenate([weights, weights]).astype(dtype=use_dtype)

    csgraph = sparse.csr_matrix(
        (weights, edges),
        shape=[
            len(vertices),
        ]
        * 2,
        dtype=use_dtype,
    )

    return csgraph




def get_path(root, target, pred):
    """
    Using a predecessor matrix from scipy.csgraph.shortest_paths, get all indices
    on the path from a root node to a target node.

    Parameters
    ----------
    root : int
        index of the root node
    target : int
        index of the target node
    pred : np.ndarray
        predecessor array from the dijkstra algorithm


    Returns
    -------
    list
        list of indices on the path from root to target

    """
    path = [target]
    p = target
    while p != root:
        p = pred[p]
        path.append(p)
    path.reverse()
    return path
