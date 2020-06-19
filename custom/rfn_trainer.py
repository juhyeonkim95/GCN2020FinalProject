#from mxnet import ndarray as nd
from mxnet import gpu

from custom.utils_rfn import make_neighborhood_matrices
from rfn.factory_functions import make_rfn, RFNLayerSpecification, FeatureInfo
from rfn.relational_fusion.normalizers import NoNormalization, L2Normalization
from mxnet.gluon.nn import ELU
from mxnet import autograd
from mxnet.gluon import Trainer
from mxnet.gluon.loss import L2Loss
from custom.trainer_interface import *


class RFNTrainer(TrainerInterface):
    def __init__(self, fusion='interactional', aggregator='attentional'):
        if fusion == 'interactional':
            name = 'rfn_int_att' if aggregator == 'attentional' else 'rfn_int_non'
        else:
            name = 'rfn_add_att' if aggregator == 'attentional' else 'rfn_add_non'
        super().__init__(name)
        self.fusion = fusion
        self.aggregator = aggregator
        self.params = None

    def save_params(self, file_name):
        self.rfn.save_parameters(file_name)

    def load_params(self, file_name):
        self.rfn.load_parameters(file_name)

    def build(self):
        target = self.train_cities if len(self.train_cities) > 0 else self.test_cities
        X_V = target[0].X_V
        X_E = target[0].X_E
        X_B = target[0].X_B
        input_feature_info = FeatureInfo.from_feature_matrices(X_V, X_E, X_B)
        print(input_feature_info)
        no_hidden_layers = 3
        hidden_layer_specs = [
            RFNLayerSpecification(
                units=16,
                fusion=self.fusion,#'interactional',
                aggregator=self.aggregator, #'attentional',
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
        self.params = self.initialize()

    def test(self, use_testset=True, verbose=True):
        loss_function = L2Loss()
        target = self.test_cities if use_testset else self.train_cities
        losses = {}
        for city in target:
            y_pred = self.rfn(city.X_V, city.X_E, city.X_B,
                              city.N_node_primal, city.N_edge_primal, city.N_mask_primal,
                              city.N_node_dual, city.N_edge_dual, city.N_common_node, city.N_mask_dual)
            loss = loss_function(y_pred, city.y).mean().asscalar()
            if verbose:
                print("Loss at City %s" % city.name, loss)
            losses[city.name] = loss
        return losses

    def initialize(self):
        self.rfn.initialize()
        params = self.rfn.collect_params()
        params.reset_ctx(ctx=gpu())
        return params

    def train(self, n_epoch, verbose=True):
        print("Training Started for RFN")
        learning_rate = 0.001
        loss_function = L2Loss()
        optimizer = 'adam'

        trainer = Trainer(self.params, optimizer, {'learning_rate': learning_rate})
        losses = []
        for epoch in range(1, n_epoch + 1):
            city = self.train_cities[epoch % len(self.train_cities)]
            with autograd.record():
                y_pred = self.rfn(city.X_V, city.X_E, city.X_B,
                                 city.N_node_primal, city.N_edge_primal, city.N_mask_primal,
                                 city.N_node_dual, city.N_edge_dual, city.N_common_node, city.N_mask_dual)
                loss = loss_function(y_pred, city.y)
            loss.backward()
            trainer.step(batch_size=len(city.y))
            if verbose:
                print(f'Loss at Epoch {epoch}: {loss.mean().asscalar()}')
            losses.append(loss.mean().asscalar())
        return losses
