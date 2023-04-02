import numpy as np
from scipy import sparse
import pytest
from GraphTEASAR import utils, graph_teasar_component, graph_teasar_all
from GraphTEASAR.utils import create_spatial_csgraph
from GraphTEASAR.find_root import find_far_points_graph


def test_create_spatial_csgraph():
    vertices = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    G = create_spatial_csgraph(vertices, edges)
    assert isinstance(G, sparse.csr_matrix)
    assert G.shape == (4, 4)

    # Check weights
    assert G[0, 1] == 1.0
    assert G[1, 2] == 1.0
    assert G[2, 3] == 1.0
    assert G[3, 0] == 1.0

    # Check for directed graph
    directed_G = create_spatial_csgraph(vertices, edges, directed=True)
    assert directed_G[0, 1] == 1.0
    assert directed_G[1, 0] == 0.0

    # Check for non-euclidean weights
    boolean_G = create_spatial_csgraph(vertices, edges, euclidean_weight=False)
    assert boolean_G[0, 1] == 1
    assert boolean_G[1, 2] == 1
    assert boolean_G[2, 3] == 1
    assert boolean_G[3, 0] == 1


def test_find_far_points_graph():
    vertices = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    G = create_spatial_csgraph(vertices, edges)

    b, a, pred, d, ds = find_far_points_graph(G)
    assert b == 0
    assert a == 2
    assert d == 2.0

    # Test with a disconnected graph
    G[3, 0] = 0.0
    G.eliminate_zeros()
    b, a, pred, d, ds = find_far_points_graph(G)
    assert b == 0
    assert a == 2
    assert d == 2.0


def test_graph_teasar_component_simple():
    vertices = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    G = create_spatial_csgraph(vertices, edges)

    paths, path_lengths = graph_teasar_component(G)
    assert len(paths) == 1
    assert len(paths[0]) == 3
    assert len(path_lengths) == 1
    assert path_lengths[0] == 2.0


def test_graph_teasar_all():
    vertices = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [2, 0], [2, 1]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]])

    G = create_spatial_csgraph(vertices, edges)

    paths, roots, path_lengths = graph_teasar_all(G, cc_vertex_thresh=0)

    # Two components with 1 path each
    assert len(paths) == 2
    assert len(path_lengths) == 2
    assert len(roots) == 2

    # Each component should have a single path of length 1
    assert path_lengths[0][0] == 2.0
    assert path_lengths[1][0] == 1.0


def test_graph_teasar_all_cc_vertex_thresh():
    vertices = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [2, 0], [2, 1]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]])

    G = utils.create_spatial_csgraph(vertices, edges)

    # Test with cc_vertex_thresh=3 (should filter out the second component)
    paths, roots, path_lengths = graph_teasar_all(G, cc_vertex_thresh=3)

    assert len(paths) == 1
    assert len(path_lengths) == 1
    assert len(roots) == 1
    assert path_lengths[0][0] == 2.0


def test_graph_teasar_all_invalidation_d():
    vertices = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 0],
            [6, 0],
            [7, 0],
            [8, 0],
            [9, 0],
            [10, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [5, -1],
            [5, -2],
            [5, -5],
        ]
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            [9, 10],
            [2, 11],
            [11, 12],
            [12, 13],
            [5, 14],
            [14, 15],
            [15, 16],
        ]
    )

    G = utils.create_spatial_csgraph(vertices, edges)

    paths, roots, path_lengths = graph_teasar_all(
        G, root_index=0, cc_vertex_thresh=0, invalidation_d=1.1
    )
    assert len(paths) == 1
    assert len(path_lengths) == 1
    assert len(path_lengths[0]) == 3
    assert len(roots) == 1
    assert roots[0] in [0, 10]
    assert path_lengths[0][0] == 10.0
    assert path_lengths[0][1] == 5.0
    assert path_lengths[0][2] == 3.0

    paths, roots, path_lengths = graph_teasar_all(
        G, root_index=0, cc_vertex_thresh=0, invalidation_d=3.1
    )
    assert len(paths) == 1
    assert len(path_lengths) == 1
    assert len(path_lengths[0]) == 2
    assert len(roots) == 1
    assert roots[0] in [0, 10]
    assert path_lengths[0][0] == 10.0
    assert path_lengths[0][1] == 5.0

    paths, roots, path_lengths = graph_teasar_all(
        G, root_index=0, cc_vertex_thresh=0, invalidation_d=5.1
    )
    assert len(paths) == 1
    assert len(path_lengths) == 1
    assert len(path_lengths[0]) == 1
    assert len(roots) == 1
    assert roots[0] in [0, 10]
    assert path_lengths[0][0] == 10.0
