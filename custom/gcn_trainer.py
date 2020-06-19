from custom.GCN import GCN, GAT, GraphSAGE
import torch.nn.functional as F
import torch
from custom.city import DGLCity
from custom.rfn_trainer import TrainerInterface


class GCNTrainer(TrainerInterface):
    def __init__(self, model_type):
        super().__init__(model_type)
        self.model_type = model_type

    def save_params(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_params(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def build(self):
        g = None
        target = self.train_cities if len(self.train_cities)  > 0 else self.test_cities
        in_size = target[0].X_E.shape[1]
        if self.model_type == 'gcn':
            self.model = GCN(g,
                             in_feats=in_size,
                             n_hidden=16,
                             n_classes=1,
                             n_layers=4,
                             activation=F.relu)
        elif self.model_type == 'graphsage':
            self.model = GraphSAGE(g,
                                   in_feats=in_size,
                                   n_hidden=16,
                                   n_classes=1,
                                   n_layers=4,
                                   activation=F.relu,
                                   dropout=0)
        elif self.model_type=='gat':
            self.model = GAT(g,
                             in_dim=in_size,
                             num_hidden=16,
                             num_classes=1,
                             num_layers=4,
                             activation=F.relu)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.cuda()

    def test(self, use_testset=True, verbose=True):
        self.model.eval()
        target = self.test_cities if use_testset else self.train_cities
        losses = {}
        for city in target:
            self.model.g = city.dual_graph
            y_pred = self.model(city.X_E).squeeze()
            loss = F.mse_loss(y_pred, city.y).item()
            if verbose:
                print("Loss at City %s" % city.name, loss)
            losses[city.name] = loss
        return losses

    def train(self, n_epoch, verbose=True):
        self.model.train()
        losses = []
        for epoch in range(1, n_epoch+1):
            city = self.train_cities[epoch % len(self.train_cities)]
            X_E = city.X_E
            y = city.y
            self.model.g = city.dual_graph
            y_pred = self.model(X_E).squeeze()
            loss = F.mse_loss(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_cpu = loss.item()
            losses.append(loss_cpu)
            if verbose:
                print('Epoch %d | Loss: %.4f' % (epoch, loss_cpu))
        return losses
