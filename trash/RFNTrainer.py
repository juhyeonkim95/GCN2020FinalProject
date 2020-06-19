#from mxnet import ndarray as nd
from mxnet import gpu

from custom.utils_rfn import make_neighborhood_matrices
from rfn.factory_functions import make_rfn, RFNLayerSpecification, FeatureInfo
from rfn.relational_fusion.normalizers import NoNormalization, L2Normalization
from mxnet.gluon.nn import ELU
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import L2Loss


class RFNTrainer:
    def __init__(self, primal_graph, dual_graph):
        self.primal_graph = primal_graph
        self.dual_graph = dual_graph

        node_indices_primal = {node: idx for idx, node in enumerate(self.primal_graph.nodes())}
        edge_indices_primal = {edge: idx for idx, edge in enumerate(self.primal_graph.edges())}

        N_node_primal, N_edge_primal, N_mask_primal = edge_neighborhoods_primal = make_neighborhood_matrices(
            self.primal_graph, node_indices_primal, edge_indices_primal)

        self.N_node_primal = N_node_primal
        self.N_edge_primal = N_edge_primal
        self.N_mask_primal = N_mask_primal

        node_indices_dual = {node: idx for idx, node in enumerate(self.dual_graph.nodes())}
        edge_indices_dual = {edge: idx for idx, edge in enumerate(self.dual_graph.edges())}

        (N_node_dual, N_edge_dual, N_mask_dual), (N_common_node, N_common_node_mask) = make_neighborhood_matrices(
            self.dual_graph, node_indices_dual, edge_indices_dual,
            is_dual=True, node_indices_primal=node_indices_primal)

        self.N_node_dual = N_node_dual
        self.N_edge_dual = N_edge_dual
        self.N_mask_dual = N_mask_dual
        self.N_common_node = N_common_node
        self.N_common_node_mask = N_common_node_mask

    def build_rfn(self, X_V, X_E, X_B):
        input_feature_info = FeatureInfo.from_feature_matrices(X_V, X_E, X_B)
        no_hidden_layers = 3
        hidden_layer_specs = [
            RFNLayerSpecification(
                units=16,
                fusion='interactional',
                aggregator='attentional',
                normalization=L2Normalization(),
                activation=ELU()
            )
            for i in range(no_hidden_layers)
        ]

        output_layer_spec = RFNLayerSpecification(
            units=1,
            fusion='additive',
            aggregator='non-attentional',
            normalization=NoNormalization(),
            activation='relu'
        )

        self.rfn = make_rfn(input_feature_info, hidden_layer_specs, output_layer_spec, output_mode='edge')

    def train(self, X_V, X_E, X_B, y, n_epoch):
        print("Training Started for RFN")
        learning_rate = 0.001
        loss_function = L2Loss()
        optimizer = 'adam'

        self.rfn.initialize()
        params = self.rfn.collect_params()
        params.reset_ctx(ctx=gpu())
        trainer = Trainer(params, optimizer, {'learning_rate': learning_rate})

        for epoch in range(1, n_epoch + 1):
            with autograd.record():
                y_pred = self.rfn(X_V, X_E, X_B,
                             self.N_node_primal, self.N_edge_primal, self.N_mask_primal,
                             self.N_node_dual, self.N_edge_dual, self.N_common_node, self.N_mask_dual)
                loss = loss_function(y_pred, y)
            loss.backward()
            trainer.step(batch_size=len(y))
            print(f'Loss at Epoch {epoch}: {loss.mean().asscalar()}')
