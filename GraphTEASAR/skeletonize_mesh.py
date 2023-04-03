"""A module for running the GraphTEASAR algorithm on spatial graphs for skeletonization of neurons."""

import numpy as np
from scipy import sparse
from . import utils
from . import graph_teasar
from functools import partial
from . import find_root


def find_neuron_root(csgraph, valid=None, is_soma_pt=None, soma_d=None):
    """function to find the root index to use for this mesh"""
    n_vertices = max(csgraph.shape[0], csgraph.shape[1])

    if valid is not None:
        temp_valid = np.copy(valid)
    else:
        temp_valid = np.ones(n_vertices, np.bool)
    assert len(temp_valid) == n_vertices

    root = None
    # soma mode
    if is_soma_pt is not None:
        # pick the first soma as root
        assert len(soma_d) == n_vertices
        assert len(is_soma_pt) == n_vertices
        is_valid_root = is_soma_pt & temp_valid
        valid_root_inds = np.where(is_valid_root)[0]
        if len(valid_root_inds) > 0:
            min_valid_root = np.nanargmin(soma_d[valid_root_inds])
            root = valid_root_inds[min_valid_root]
            root_ds, pred = sparse.csgraph.dijkstra(
                csgraph, directed=False, indices=root, return_predecessors=True
            )
        else:

            root, pred, root_ds, temp_valid = find_root.find_graph_root(
                csgraph, valid=temp_valid
            )
        temp_valid[is_soma_pt] = False

    if root is None:
        # there is no soma close, so use far point heuristic
        root, _, pred, _, root_ds = find_root.find_far_points_graph(
            csgraph, valid=temp_valid
        )
    temp_valid[root] = False
    assert np.all(~np.isinf(root_ds[temp_valid]))
    return root, root_ds, pred, temp_valid


def skeletonize_mesh(
    vertices: np.ndarray,
    edges: np.ndarray = None,
    faces: np.ndarray = None,
    invalidation_d: float = 10000,
    cc_vertex_thresh: int = 0,
    return_map: bool = False,
    root_func: callable = find_root.find_graph_root,
    root_index: int = None,
    progress:bool =True
):
    """Skeletonizes a mesh by first finding the connected components of the mesh
    and then skeletonizing each connected component. The connected components are
    found by finding the connected components of the graph of the mesh.

    Parameters
    ----------
    vertices : np.ndarray
        Nx3 array of vertex positions
    edges : np.ndarray, optional
        Mx2 array of edge indices, by default None
        (must pass one of edges, faces or csgraph)
    faces : np.ndarray, optional
        Kx3 array of face indices, by default None
        (must pass one of edges, faces or csgraph)
    invalidation_d : float, optional
        distance threshold for invalidating edges, by default 10000
    cc_vertex_thresh : int, optional
        minimum number of vertices for a connected component to be skeletonized, by default 0
    return_map : bool, optional
        whether to return the map of vertex indices to skeleton indices, by default False
    root_index : int, optional
        index of root vertex, by default None
    progress : bool, optional
        whether to show a progress bar, by default True

    Returns
    -------
    all_paths: C long list of lists of vertex indices
        each list represents a component,
        within each component there is a list of cover paths
        where each path is a list of the vertices visited by that path
        starting away from root and heading toward the root
        ending on either the root, or a previously visiting vertex
    roots: A list of length C of indices
        the index of the root vertex for each of the C components
    tot_path_lengths: A list of length C of floats
        the total length of the paths for each of the C components
    mesh_to_skeleton_map: array of vertex indices (only if return_map is True)
        the index of the vertex on the skeleton that is closest to each vertex in the mesh
    """
    if faces is not None:
        if edges is not None:
            raise ValueError("Pass one of faces or edges")
        else:
            edges = utils.faces_to_edges(faces)
    csgraph = utils.create_spatial_csgraph(vertices, edges)


    return_vals = graph_teasar(
        csgraph,
        root_index=root_index,
        root_func=root_func,
        invalidation_d=invalidation_d,
        cc_vertex_thresh=cc_vertex_thresh,
        return_map=return_map,
        progress=progress
    )
    return return_vals


def skeletonize_neuron(
    vertices,
    edges: np.ndarray = None,
    faces: np.ndarray = None,
    root_index: int = None,
    soma_pt: np.array = None,
    soma_thresh: float = 10000,
    invalidation_d: float = 10000,
    cc_vertex_thresh: int = 100,
    return_map: bool = False,
    progress:bool =True
):
    """skeletonization routine for skeletonizing a neuron
    makes extra assumptions about how you want to handle to handle root finding
    and soma points, making sure that the soma is always the root
    and that any points that are within the soma_thresh of the soma are not skeletonized

    Note: if your mesh is not a single connected component, then this will skeletonize
    each connected component separately, and the paths will not form a single connected component

    If you are expecting a single component, you must ensure that you have filtered your
    mesh to only include a single component, or set the cc_vertex_thresh large enough
    that there is only one component with at least that many vertices.

    Parameters
    ----------
    vertices : np.ndarray
        NxK (usually Nx3) array of vertex positions
    edges : np.ndarray, optional
        Mx2 array of edge indices, by default None
        (must pass one of edges or faces)
    faces : np.ndarray, optional
        Kx3 array of face indices, by default None
        (must pass one of edges or faces)
    root_index : int, optional
        index of root vertex, by default None
        If you have precalcualted which vertex you want to be the root, pass it here
        otherwise it will rely upon the soma_pt to find the closest vertex for the soma
        If you don't pass one it will use the soma_pt
    soma_pt : np.array, optional
        1xK array of soma position, by default None
        If you don't pass one it will find the mutual farthest point pair in the graph
        (and repeat this for each connected component)
    soma_thresh : float, optional
        distance threshold for soma points, by default 10000
        (in same units as vertices, so default assumes nanometers)
    invalidation_d : float, optional
        distance threshold for invalidating edges, by default 10000
        (in same units as vertices, so default assumes nanometers)
    cc_vertex_thresh : int, optional
        minimum number of vertices for a connected component to be skeletonized, by default 100
        (units is integers, so will vary depending on graph resolution)
    return_map : bool, optional
        whether to return the map of vertex indices to skeleton indices, by default False
    progress : bool, optional
        whether to show a progress bar, by default True

    Returns
    -------
    all_paths: C long list of lists of vertex indices
        each list represents a component,
        within each component there is a list of cover paths
        where each path is a list of the vertices visited by that path
        starting away from root and heading toward the root
        ending on either the root, or a previously visiting vertex
    roots: A list of length C of indices
        the index of the root vertex for each of the C components
    tot_path_lengths: A list of length C of floats
        the total length of the paths for each of the C components
    mesh_to_skeleton_map: array of vertex indices (only if return_map is True)
        the index of the vertex on the skeleton that is closest to each vertex in the mesh
    """

    n_vertices = vertices.shape[0]
    if root_index is not None:
        soma_d = np.linalg.norm(vertices - vertices[root_index], axis=1)
        is_soma_pt = np.arange(n_vertices) == root_index
    elif soma_pt is not None:
        soma_d = vertices - soma_pt.reshape(1, vertices.shape[1])
        soma_d = np.linalg.norm(soma_d, axis=1)
        is_soma_pt = soma_d < soma_thresh
    else:
        is_soma_pt = None
        soma_d = None
    # is_soma_pt = None
    # soma_d = None

    root_func = partial(
        find_root.find_neuron_root, is_soma_pt=is_soma_pt, soma_d=soma_d
    )
    return_vals = skeletonize_mesh(
        vertices,
        edges,
        faces,
        invalidation_d=invalidation_d,
        cc_vertex_thresh=cc_vertex_thresh,
        return_map=return_map,
        root_func=root_func,
        root_index=root_index,
        progress=progress
    )
    return return_vals
