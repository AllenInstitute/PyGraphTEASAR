from scipy import sparse
import numpy as np
import time
from . import utils
from . import find_root
from tqdm import tqdm


__version__ = "0.1.1"


def graph_teasar_component(
    csgraph,
    root=None,
    valid=None,
    root_ds=None,
    root_pred=None,
    find_root_func=find_root.find_graph_root,
    invalidation_d=10000,
    return_timing=False,
    return_map=False,
    progress: bool = True,
):
    """core skeletonization function used to skeletonize a single component of a mesh

    Parameters
    ----------
    csgraph : scipy.sparse.csr_matrix
        sparse matrix of the graph
    root : int, optional
        index of the starting point, by default None
    valid : np.array, optional
        array of valid vertices, by default None
    root_ds : np.array, optional
        distance array from the dijkstra algorithm for the root, by default None
    root_pred : np.array, optional
        predecessor array from the dijkstra algorithm for the root, by default None
    find_root_func : function, optional
        function to find the root, by default utils.find_graph_root
        needs to return
    invalidation_d : int, optional
        distance to invalidate vertices, by default 10000
    return_timing : bool, optional
        whether to return timing information, by default False
    return_map : bool, optional
        whether to return the distance map, by default False
    progress : bool, optional
        whether to show a progress bar, by default True

    Returns
    -------
    list
        list of paths
    list
        list of path lengths
    np.array
        distance map
    np.array
        map of which vertex each vertex is mapped to (only if return_map is True)
    list
        list of timing information (only if return_timing is True)

    Raises
    ------
    ValueError
        if valid is not None and not the same length as the number of vertices
    ValueError
        if all valid vertices are not reachable from the root

    """
    n_vertices = max(csgraph.shape[0], csgraph.shape[1])
    # if no root passed, then calculation one
    if root is None:
        if valid is not None:
            (root, root_pred, root_ds, valid) = find_root_func(csgraph, valid=valid)
        else:
            (root, root_pred, root_ds, valid) = find_root_func(csgraph)
    # if root_ds have not be precalculated do so
    if root_ds is None:
        root_ds, root_pred = sparse.csgraph.dijkstra(
            csgraph, False, root, return_predecessors=True
        )
    # if certain vertices haven't been pre-invalidated start with just
    # the root vertex invalidated
    if valid is None:
        valid = np.ones(n_vertices, bool)
        valid[root] = False
    else:
        if len(valid) != n_vertices:
            raise ValueError("valid must be length of vertices")

    if return_map == True:
        mesh_to_skeleton_dist = np.full(n_vertices, np.inf)
        mesh_to_skeleton_map = np.full(n_vertices, np.nan)

    total_to_visit = np.sum(valid)
    if np.sum(np.isinf(root_ds) & valid) != 0:
        print(np.where(np.isinf(root_ds) & valid))
        raise ValueError("all valid vertices should be reachable from root")

    # vector to store each branch result
    paths = []

    # vector to store each path's total length
    path_lengths = []

    # keep track of the nodes that have been visited
    visited_nodes = [root]

    # counter to track how many branches have been counted
    k = 0

    # arrays to track timing
    start = time.time()
    time_arrays = [[], [], [], [], []]

    with tqdm(total=total_to_visit, disable=not progress) as pbar:
        # keep looping till all vertices have been invalidated
        while np.sum(valid) > 0:
            k += 1
            t = time.time()
            # find the next target, farthest vertex from root
            # that has not been invalidated
            target = np.nanargmax(root_ds * valid)
            if np.isinf(root_ds[target]):
                raise ValueError("target cannot be reached")
            time_arrays[0].append(time.time() - t)

            t = time.time()
            # figure out the longest this branch could be
            # by following the route from target to the root
            # and finding the first already visited node (max_branch)
            # The dist(root->target) - dist(root->max_branch)
            # is the maximum distance the shortest route to a branch
            # point from the target could possibly be,
            # use this bound to reduce the djisktra search radius for this target
            max_branch = target
            while max_branch not in visited_nodes:
                max_branch = root_pred[max_branch]
            max_path_length = root_ds[target] - root_ds[max_branch]

            # calculate the shortest path to that vertex
            # from all other vertices
            # up till the distance to the root
            ds, pred_t = sparse.csgraph.dijkstra(
                csgraph,
                False,
                target,
                limit=max_path_length,
                return_predecessors=True,
            )

            # pick out the vertex that has already been visited
            # which has the shortest path to target
            min_node = np.argmin(ds[visited_nodes])
            # reindex to get its absolute index
            branch = visited_nodes[min_node]
            # this is in the index of the point on the skeleton
            # we want this branch to connect to
            time_arrays[1].append(time.time() - t)

            t = time.time()
            # get the path from the target to branch point
            path = utils.get_path(target, branch, pred_t)
            visited_nodes += path[0:-1]
            # record its length
            assert ~np.isinf(ds[branch])
            path_lengths.append(ds[branch])
            # record the path
            paths.append(path)
            time_arrays[2].append(time.time() - t)

            t = time.time()
            # get the distance to all points along the new path
            # within the invalidation distance
            dm, _, sources = sparse.csgraph.dijkstra(
                csgraph,
                False,
                path,
                limit=invalidation_d,
                min_only=True,
                return_predecessors=True,
            )
            time_arrays[3].append(time.time() - t)

            t = time.time()
            # all such non infinite distances are within the invalidation
            # zone and should be marked invalid
            nodes_to_update = ~np.isinf(dm)
            marked = np.sum(valid & ~np.isinf(dm))
            if return_map == True:
                new_sources_closer = (
                    dm[nodes_to_update] < mesh_to_skeleton_dist[nodes_to_update]
                )
                mesh_to_skeleton_map[nodes_to_update] = np.where(
                    new_sources_closer,
                    sources[nodes_to_update],
                    mesh_to_skeleton_map[nodes_to_update],
                )
                mesh_to_skeleton_dist[nodes_to_update] = np.where(
                    new_sources_closer,
                    dm[nodes_to_update],
                    mesh_to_skeleton_dist[nodes_to_update],
                )

            valid[~np.isinf(dm)] = False

            # print out how many vertices are still valid
            pbar.update(marked)
            time_arrays[4].append(time.time() - t)
    # record the total time
    dt = time.time() - start

    out_tuple = (paths, path_lengths)
    if return_map:
        out_tuple = out_tuple + (mesh_to_skeleton_map,)
    if return_timing:
        out_tuple = out_tuple + (time_arrays, dt)

    return out_tuple


def graph_teasar(
    csgraph,
    root_index=None,
    root_function=find_root.find_graph_root,
    invalidation_d=10000,
    cc_vertex_thresh=0,
    return_map=False,
    progress: bool = True,
):
    """skeletonize a mesh, seperately for each connected component in the mesh above a threshold

    Parameters
    ----------
    csgraph : scipy.sparse.csr_matrix
        sparse matrix of the mesh, with edge weights as the distance between vertices
        of size NxN
    root_index : int, optional
        index of the root vertex, by default None
    root_function : function, optional
        function to find the root vertex, by default find_root.find_graph_root
        can be any function which takes a csgraph and a valid boolean array
        This function will be called for each component mask of the graph
        with the valid array set for as a mask of that component.

        The function needs to select a root for that component, and return the following tuple.
            root: int
                the index of the root for this component
            pred: np.ndarray
                a N long array of predecessors that lead back to root (or None and dijkstra will be used to calculate it)
            root_ds: np.ndarray
                a N long array of distances from root (or None and dijkstra will be used to calculate it)
            valid: np.array
                a N long mask of pre-invalidated vertices for this component
                can be the same as the input valid array if no vertices are pre-invalidated
    invalidation_d : int, optional
        distance to invalidate vertices, by default 10000
        units in the same as the edge weights of the csgraph
        default is tuned for nanometers and neuron mesh graphs
    cc_vertex_thresh : int, optional
        minimum number of vertices in a connected component to skeletonize, by default 0
    return_map : bool, optional
        whether to return the map from mesh vertices to skeleton vertices, by default False
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

    # find all the connected components in the mesh
    n_vertices = max(csgraph.shape[0], csgraph.shape[1])
    n_components, labels = sparse.csgraph.connected_components(
        csgraph, directed=False, return_labels=True
    )
    _, comp_counts = np.unique(labels, return_counts=True)

    if return_map:
        mesh_to_skeleton_map = np.full(n_vertices, np.nan)

    # variables to collect the paths, roots and path lengths
    all_paths = []
    roots = []
    tot_path_lengths = []

    # is_soma_pt = None
    # soma_d = None

    # loop over the components
    for k in range(n_components):
        if comp_counts[k] > cc_vertex_thresh:

            # if root_index is not None and its in this component
            # then use it as root
            if root_index is not None and labels[root_index] == k:
                root = root_index
                root_ds, pred = sparse.csgraph.dijkstra(
                    csgraph, False, root, return_predecessors=True
                )
                valid = labels == k
            else:
                # otherwise find the root using the root function
                # and the connected component
                root, pred, root_ds, valid = root_function(csgraph, labels == k)
            valid[root] = False

            # run teasar on this component
            teasar_output = graph_teasar_component(
                csgraph,
                root=root,
                root_ds=root_ds,
                root_pred=pred,
                valid=valid,
                invalidation_d=invalidation_d,
                return_map=return_map,
                progress=progress,
            )
            if return_map is False:
                paths, path_lengths = teasar_output
            else:
                paths, path_lengths, mesh_to_skeleton_map_single = teasar_output
                mesh_to_skeleton_map[
                    ~np.isnan(mesh_to_skeleton_map_single)
                ] = mesh_to_skeleton_map_single[~np.isnan(mesh_to_skeleton_map_single)]

            if len(path_lengths) > 0:
                # collect the results in lists
                tot_path_lengths.append(path_lengths)
                all_paths.append(paths)
                roots.append(root)

    if return_map:
        return all_paths, roots, tot_path_lengths, mesh_to_skeleton_map
    else:
        return all_paths, roots, tot_path_lengths
