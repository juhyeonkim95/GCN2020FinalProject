class TrainerInterface:
    def __init__(self, name):
        self.train_cities = []
        self.test_cities = []
        self.name = name

    def build(self):
        pass

    def train(self, n_epoch, verbose=True):
        pass

    def test(self, verbose=True):
        pass

    def add_city(self, city, train=True):
        if train:
            self.train_cities.append(city)
        else:
            self.test_cities.append(city)