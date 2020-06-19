import numpy as np
from mxnet import nd, gpu


def make_neighborhood_matrices(graph, node_indices, edge_indices,
                               is_dual=False, node_indices_primal=None):
    """
        Converts a graph into node and edge the required neighborhood matrices with asso

        Args:
            graph: A networkx primal or dual graph presentation of a road network.
            node_indices: A map from node objects in the graph to node indexes in
                          the output node neighborhood matrix N_node.
            edge_indices: A map from edge objects in the graph to edge indexes in
                          the output edge neighborhood matrix N_edge.
            is_dual: Boolean flag that indicates whether the graph is a dual graph.
            node_indices_primal: Must be supplied if is_dual is True.
                                 Maps node objects in the primal graph to node indexes in
                                 the common node neighborhood matrix N_common_node.
        Returns:
            N_node: A node adjacency list in matrix format.
                    The ith row contains the node indices of the nodes
                    in the neighborhood of the ith node. of the ith node.
            N_edge: A node adjacency list in matrix format.
                    The ith row contains the edge indices of the edges connecting the ith
                    node to its neighbors.
            N_mask: A matrix that indicates whether the jth entry in N_node or N_edge
                    exists in the graph.
            N_common_node: Only returned if is_dual=True.
                           A common node is a node that is common to the two edges
                           connected by a between-edge. The ith row and jth column in
                           this matrix contains the index of the common node that connects
                           ith edge (a node in the dual graph) is connected to its jth neighbor.
            N_common_mask: Only returned if is_dual=True.
                           Similar to N_mask, N_common_mask indicates whether the jth entry in
                           N_common_node exists in the graph.

        Raises:
            KeyError: Raises an exception.
    """
    assert not is_dual or is_dual and node_indices_primal is not None
    N_node = []
    N_edge = []

    nodes = sorted(
        list(graph.nodes()),
        key=lambda node: node_indices[node])

    for node in nodes:
        node_neighbors = []
        edge_neighbors = []

        predecessors = graph.predecessors(node)
        for neighbor in predecessors:
            node_neighbors.append(node_indices[neighbor])
            edge = (neighbor, node)
            edge_neighbors.append(edge_indices[edge])

        successors = graph.successors(node)
        for neighbor in successors:
            node_neighbors.append(node_indices[neighbor])
            edge = (node, neighbor)
            edge_neighbors.append(edge_indices[edge])

        assert len(node_neighbors) == len(edge_neighbors)
        N_node.append(node_neighbors)
        N_edge.append(edge_neighbors)

    N_node, N_mask = mask_neighborhoods(N_node)
    N_edge, _ = mask_neighborhoods(N_edge)
    N_mask = N_mask.reshape(*N_node.shape[:2], 1)

    if is_dual:
        N_common_node = [
            [node_indices_primal[edge[0][1]]]
            for edge in graph.edges()]
        N_common_node, N_common_node_mask = mask_neighborhoods(N_common_node, is_dual)
        N_common_node_mask = N_common_node_mask.reshape(*N_common_node.shape[:2], 1)
        return (N_node, N_edge, N_mask), (N_common_node, N_common_node_mask)
    else:
        return N_node, N_edge, N_mask


def mask_neighborhoods(neighborhoods_list, is_dual=False):
    max_no_neighbors = (
        max(len(n) for n in neighborhoods_list) if not is_dual
        else 1)
    shape = (len(neighborhoods_list), max_no_neighbors)

    neighborhoods_array = nd.zeros(
        shape=shape,
        dtype=np.int32,
        ctx=gpu())
    mask = nd.zeros(shape=shape, ctx=gpu())

    for idx, neighborhood in enumerate(neighborhoods_list):
        neighborhood_size = len(neighborhood)
        if neighborhood_size == 0:
            mask[:] = 1
            continue
        else:
            neighborhoods_array[idx][:neighborhood_size] = neighborhood
            mask[idx][:neighborhood_size] = 1

    return neighborhoods_array, mask