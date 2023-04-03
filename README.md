# GraphTEASAR

GraphTEASAR is a Python library for running the TEASAR algorithm on spatial graphs for skeletonization. It provides a graph generalization of the TEASAR algorithm, which was originally designed to run on voxelized data but can be applied to any abstract graph with weights.

GraphTEASAR implements a version of the basic core TEASAR approach on abstract graphs and provides functions to make it easy to run the algorithm on spatial graphs, such as meshes, where the vertices represent points in space and the weights are the Euclidean distance between vertices. The TEASAR algorithm defined alternative distance metrics within a graph to drive the cost of traversing the graph towards the center of the object, resulting in an accurate and robust skeletonization. GraphTEASAR operates on the spatial graph, and so runs along the outside of the object.

# Installation
To install GraphTEASAR, you can use pip:

    pip install GraphTEASAR

Or you can clone the repository and install it locally:

    git clone https://github.com/AllenInstitute/PyGraphTEASAR.git
    cd PyGraphTEASAR
    pip install .

# Usage
GraphTEASAR provides functions for running the algorithm on different types of data. Depending on your problem, you might want to interface with the core graph TEASAR directly or use the convenience functions provided.

The documentation provides an introduction to the different levels you might interface with the algorithm, depending on what type of data you are working with and what your ultimate application is.

## References
Sato, M., Bitter, I., Bender, M. A., Kaufman, A. E., & Nakajima, M. (n.d.). TEASAR: tree-structure extraction algorithm for accurate and robust skeletons. In Proceedings the Eighth Pacific Conference on Computer Graphics and Applications. IEEE Comput. Soc. https://doi.org/10.1109/pccga.2000.883951

## Contributing
If you find a bug or have a feature request, please create an issue on the GitHub repository. If you would like to contribute code, please fork the repository and submit a pull request.

## License
GraphTEASAR is released under the Allen Institute Software License. See the LICENSE file for details.