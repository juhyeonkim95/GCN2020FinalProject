from custom.trainer_interface import TrainerInterface
import torch
import torch.nn.functional as F


class MLPTrainer(TrainerInterface):
    def __init__(self):
        super().__init__("mlp")

    def save_params(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_params(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def build(self):
        target = self.train_cities if len(self.train_cities) > 0 else self.test_cities
        in_size = target[0].X_E.shape[1]
        self.model = MLP(in_feats=in_size,
                         n_hidden=16,
                         n_classes=1,
                         n_layers=4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.cuda()

    def train(self, n_epoch, verbose=True):
        self.model.train()
        losses = []
        for epoch in range(1, n_epoch + 1):

            city = self.train_cities[epoch % len(self.train_cities)]
            X_E = city.X_E
            y = city.y

            batch_size = 256
            index = 0
            while True:
                X_E_batch = X_E[index:index + batch_size]
                y_batch = y[index:index + batch_size]
                y_pred = self.model(X_E_batch).squeeze()

                loss = F.mse_loss(y_pred, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_cpu = loss.item()

                index += batch_size
                if index > X_E.shape[0]:
                    losses.append(loss_cpu)
                    break

        return losses

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


class MLP(torch.nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        # input layer
        self.layers.append(torch.nn.Linear(in_feats, n_hidden))
        #self.layers.append(torch.nn.ReLU())
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(torch.nn.Linear(n_hidden, n_hidden))
            #self.layers.append(torch.nn.ReLU())
        # output layer
        self.layers.append(torch.nn.Linear(n_hidden, n_classes))
        #self.layers.append(torch.nn.ReLU())
        self.act = torch.nn.ReLU()

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = self.act(h)
        return h