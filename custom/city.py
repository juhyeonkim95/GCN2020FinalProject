from custom.utils_rfn import *

class City:
    pass

class RFNCity(City):
    def __init__(self, name, primal_graph, dual_graph):
        self.name = name

        node_indices_primal = {node: idx for idx, node in enumerate(primal_graph.nodes())}
        edge_indices_primal = {edge: idx for idx, edge in enumerate(primal_graph.edges())}

        N_node_primal, N_edge_primal, N_mask_primal = edge_neighborhoods_primal = make_neighborhood_matrices(
            primal_graph, node_indices_primal, edge_indices_primal)

        self.N_node_primal = N_node_primal
        self.N_edge_primal = N_edge_primal
        self.N_mask_primal = N_mask_primal

        node_indices_dual = {node: idx for idx, node in enumerate(dual_graph.nodes())}
        edge_indices_dual = {edge: idx for idx, edge in enumerate(dual_graph.edges())}

        (N_node_dual, N_edge_dual, N_mask_dual), (N_common_node, N_common_node_mask) = make_neighborhood_matrices(
            dual_graph, node_indices_dual, edge_indices_dual,
            is_dual=True, node_indices_primal=node_indices_primal)

        self.N_node_dual = N_node_dual
        self.N_edge_dual = N_edge_dual
        self.N_mask_dual = N_mask_dual
        self.N_common_node = N_common_node
        self.N_common_node_mask = N_common_node_mask

    def set_features(self, X_V, X_E, X_B, y):
        self.X_V = X_V
        self.X_E = X_E
        self.X_B = X_B
        self.y = y


class DGLCity(City):
    def __init__(self, name, dual_graph):
        self.name = name
        self.dual_graph = dual_graph

    def set_features(self, X_E, y):
        self.X_E = X_E.cuda()
        self.y = y.cuda()
