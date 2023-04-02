.. title:: Introduction 

Introduction
************

TEASAR
------
This package is designed to implement a graph generalization of the TEASAR algorithm, first published by Sato and collegues [1]_

Their original algorithm was imagined to run on a voxelized data, but the concepts can 
apply to any abstract graph with weights. Voxel data can also be thought of as a graph,
where neighboring voxels are connected with a weight or distance of 1. 

The TEASAR algorithm in fact defined alternative distances metrics within the voxel graph, 
to force the cost of traversing voxels to drive the skeletonizaton into the center of the object. 

This package implements a version of the basic core TEASAR approach on abstract graphs.

It also provides functions to make it easy to run the algorithm on spatial graphs, such as meshes, where the vertices represents points in space and the weights are the euclidean distance between vertices. 

Finally, it provides further conveince functions for running the algorithm on meshes of neurons, where special care to define the root node and to collapse the soma is taken.

Depending on what problem you are facing, you might want to interface with the core graph TEASAR directly, or you might want to use the convenience functions provided here.

This guide will introduce you to the different levels you might interface with the algorithm, depending on what type of data you are working with and what your ultimate application is.  It will work from the bottom up, describing the simplest most core functions first, and then building up to the most complex and specialized.

.. [1] Sato, M., Bitter, I., Bender, M. A., Kaufman, A. E., & Nakajima, M. (n.d.). TEASAR: tree-structure extraction algorithm for accurate and robust skeletons. In Proceedings the Eighth Pacific Conference on Computer Graphics and Applications. IEEE Comput. Soc. https://doi.org/10.1109/pccga.2000.883951

GraphTEASAR
-----------
The most basic function provided is a function to skeletonize a scipy csgraph object, 
where weights =0 are considered to be disconnected vertices, and weights>0 reflect the cost of traversing that edge.  The lowest level function :func:`GraphTEASAR.graph_teasar_component` assumes that the graph is fully connected, and will raise a ValueError if this is not true.

Example
::

    from GraphTEASAR import graph_teasar_component

    paths, path_distances, vertex_map = graph_teasar_component(graph, 
                                invalidation_d=10,
                                return_map=True)

The other arguments are optional and generally help if you have already calculated some of the information needed to run the algorithm.  They are described in the docstring. What is returned is a list of the paths that the algorithm found, along with the total path length of each path.  Optionally you can return a map of which vertices were invalidated by which vertex on the path. This essentially provides a way to associated each node of the graph with a node of the skeleton. 

The next most complex function is :func:`GraphTEASAR.graph_teasar_all` which will skeletonize a graph that may have multiple connected components. It will skeletonize each component separately, and return all the results.  The find_root function that is passed will be passed a valid mask for each of the connected components, so it can know what vertices to be selecting from when picking a root for each component. 

Example
::

    from GraphTEASAR import graph_teasar_all

    output = graph_teasar_all(graph,
                              invalidation_d=10,
                              cc_vertex_threshold=10,
                              return_map=True)
    all_paths, roots, all_path_distances, vertex_map = output

The all_paths are a list of lists, where each entry in the outer list is a connected component above the cc_vertex_threshold.  The roots are the root nodes that were selected for each component.  The all_path_distances are the total path lengths for each path in each component.  The vertex_map is a list of arrays, where each array is a map of which vertices were invalidated by which vertex on the path for each component.

Spatial graphs
--------------
A natural application of the graph TEASAR algorithm is to run it on spatial graphs, or 'meshes' where the vertices are points in space and there are some edges between them, and the weights are the euclidean distance between vertices.  So if you have a set of vertices and edges or faces between those vertices, this package provides a convenience module (:mod:`GraphTEASAR.skeletonize_mesh`) for you to simplify creating skeletons on such objects. It has functions which take in a mesh as a set of vertices, and either a set of edges, a set of faces, or a csgraph, and then runs the graph TEASAR algorithm on the mesh.  

Example
::

    from GraphTEASAR import skeletonize_mesh

    output = skeletonize_mesh.skeletonize_mesh(vertices, faces, invalidation_d=10,
                              return_map=True)
    paths, path_distances, vertex_map = output

It will also automatically find the root node for each component, and collapse the soma if it is present.  It will also return the paths in the original mesh coordinates, rather than the graph coordinates.

This approach was first developed to skeletonize meshes of neurons, and so there are some special heuristic approaches to root choosing and invalidation based on where the soma of the neuron is.  As such we provide the function :func:`GraphTEASAR.skeletonize_mesh.skeletonize_neuron` which has arguments to specify the soma position and a soma radius.  These will be used to create a partial function version of the :func:`GraphTEASAR.find_root.find_neuron_root` , and collapse the soma if it is present (root finding discussed below).  It will also return the paths in the original mesh coordinates, rather than the graph coordinates.

Post-Processing
---------------
Sometimes you 
Algorithm
---------
The algorithm at it's core works on a connected component of the mesh graph.
Disconnected components are skeletonized separately, and trivially combined.

For each component, first a root node is found and a valid mask the describes the set of nodes that this skeletonization needs to visit is initialized. For many applications this is simply the set of vertices in this component.  A discussion of root node finding and initialization of the validation mask is below.

Then a while loop is entered. Within the loop, first, the farthest still valid target along the graph from the root node is found,
and the shortest path along the graph is drawn from target to existing skeleton paths and added to the skeleton.

Second, nodes that are within the parameterized distance :obj:`invalidation_d` ALONG THE  GRAPH from that new skeleton path are invalidated.
If :obj:`return_map` is selected, the algorithm will remember which skeleton path vertex was responsible for invalidating each mesh vertex.

This loop continues until all vertices are invalidated, and because we analyze one connected component at a time this is guaranteed to finish.  

The resulting algorithm does not incorporate all aspects of the original TEASAR algorithm, namely it does not explicitly try to change the cost of traversing the graph in order to bias paths to travel in the center of the object.  Instead, the problem of determining the correct distance function to use between nodes is left up to the constructor of the graph. A natural distance metric is to use the euclidean distance between nodes, when those nodes reflect vertices in space, but modifications of that metric could be considered.

---------------------------------
root finding and validation masks
---------------------------------
The algorithm requires that a root node be selected, and so one must provide a function to the algorithm to select the root.  In addition, some use cases might desire to invalidate some of the nodes initially, so that skeletons paths are never drawn to those nodes. For example, in the case of skeletonizing a mesh of a neuron, one might not want any skeleton paths to have to visit the nodes near the some of that neuron. If you didn't do this, than the skeletonization routine produce many minor branches along the surface of the soma. Other graphs may have other heuristics for finding roots and doing an initial invalidation of nodes.

As a general heuristic on graphs, we provide a default function to find a root node.  This function is :func:`GraphTEASAR.find_root.find_graph_root`.  It simply starts with a random vertex in the component, and then find the farther target from that index, then the farthest target from that target, and so on, until the next target is no longer any farther away that the previous one. This places the root at an extreme end of the object.

Alternative functions needs to take in two arguments, the csgraph being evaluated and binary mask indicating the set of vertices that the root should be chosen amongst. They should return 4 values, including the root_index selected, as well as an array of predecessors that describe how to get to the root, the distances to the root, and an validation mask that describes the set of vertices that must be visited by the skeltonization routine. 

We provide an example function :func:`GraphTEASAR.find_root.find_neuron_root` which is designed to be turned into a partial function using `functools.partial` given a soma_pt and a soma_radius.  This will pick a root that is closest to the soma_pt and inside the radius, or fall back to the find_graph_root method if the component has no points near the soma_pt. It also pre-invalidates all points close to the soma. 

This is the function utilized by the :func:`GraphTEASAR.skeletonize_mesh.skeletonize_neuron` function described below.

