from custom.GCN import GCN, GAT, GraphSAGE
import torch.nn.functional as F
import torch


class GCNTrainer:
    def __init__(self, g, in_size, model_type):
        if model_type == 'gcn':
            self.model = GCN(g,
                             in_feats=in_size,
                             n_hidden=8,
                             n_classes=1,
                             n_layers=4,
                             activation=F.relu)
        elif model_type == 'graphsage':
            self.model = GraphSAGE(g,
                                   in_feats=in_size,
                                   n_hidden=8,
                                   n_classes=1,
                                   n_layers=4,
                                   activation=F.relu,
                                   dropout=0)
        elif model_type=='gat':
            self.model = GAT(g,
                             in_dim=in_size,
                             num_hidden=8,
                             num_classes=1,
                             num_layers=4,
                             activation=F.relu)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.cuda()
        self.model.train()

    def train(self, X, Y, n_epoch):
        for epoch in range(1, n_epoch+1):
            X = X.cuda()
            Y = Y.cuda()
            Y_pred = self.model(X).squeeze()
            loss = F.mse_loss(Y_pred, Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
