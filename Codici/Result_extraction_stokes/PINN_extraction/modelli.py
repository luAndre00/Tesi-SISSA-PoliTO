
import torch
import matplotlib.pyplot as plt
from pina import LabelTensor
from pina.model import FeedForward
from pina.model.layers import FourierFeatureEmbedding

class Strong_net(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = FeedForward(*args, **kwargs)

    def forward(self, x):
        out = self.model(x)
        out.labels = ['vx', 'vy', 'p', 'ux', 'uy', 'r', 'zx', 'zy']
        p = LabelTensor(out.extract(['p']), 'p')
        r = LabelTensor(out.extract(['r']), 'r')
        vx = LabelTensor(x.extract(['y']) + out.extract(['vx']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['vx'])
        vy = LabelTensor(out.extract(['vy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['vy'])
        ux = LabelTensor(out.extract(['ux']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['ux'])
        uy = LabelTensor(out.extract(['uy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['uy'])
        zx = LabelTensor(out.extract(['zx']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['zx'])
        zy = LabelTensor(out.extract(['zy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['zy'])
        return vx.append(vy).append(p).append(ux).append(uy).append(r).append(zx).append(zy)

class MultiscaleFourierNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embedding1 = FourierFeatureEmbedding(input_dimension=3, output_dimension=100, sigma=1)
        self.embedding2 = FourierFeatureEmbedding(input_dimension=3, output_dimension=100, sigma=10)
        self.layers = FeedForward(*args, **kwargs)
        self.final_layer = torch.nn.Linear(2*100, 8)

    def forward(self, x):
        e1 = self.layers(self.embedding1(x))
        e2 = self.layers(self.embedding2(x))
        out = self.final_layer(torch.cat([e1, e2], dim=-1))
        out.labels = ['vx', 'vy', 'p', 'ux', 'uy', 'r', 'zx', 'zy']
        return out

class MultiscaleFourierNet_strong(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embedding1 = FourierFeatureEmbedding(input_dimension=3, output_dimension=100, sigma=1)
        self.embedding2 = FourierFeatureEmbedding(input_dimension=3, output_dimension=100, sigma=10)
        self.layers = FeedForward(*args, **kwargs)
        self.final_layer = torch.nn.Linear(2*100, 8)

    def forward(self, x):
        e1 = self.layers(self.embedding1(x))
        e2 = self.layers(self.embedding2(x))
        out = self.final_layer(torch.cat([e1, e2], dim=-1))
        out.labels = ['vx', 'vy', 'p', 'ux', 'uy', 'r', 'zx', 'zy']
        p = LabelTensor(out.extract(['p']), 'p')
        r = LabelTensor(out.extract(['r']), 'r')
        vx = LabelTensor(x.extract(['y']) + out.extract(['vx']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['vx'])
        vy = LabelTensor(out.extract(['vy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['vy'])
        ux = LabelTensor(out.extract(['ux']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['ux'])
        uy = LabelTensor(out.extract(['uy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['uy'])
        zx = LabelTensor(out.extract(['zx']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['zx'])
        zy = LabelTensor(out.extract(['zy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['zy'])
        return vx.append(vy).append(p).append(ux).append(uy).append(r).append(zx).append(zy)
