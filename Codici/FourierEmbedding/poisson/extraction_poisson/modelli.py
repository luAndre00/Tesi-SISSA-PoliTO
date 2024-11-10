
import torch
import matplotlib.pyplot as plt
from pina import LabelTensor
from pina.model import FeedForward
from pina.model.layers import FourierFeatureEmbedding

alpha = 0.008

# Nuovo tipo di modello (fatto da Dario)
class MultiscaleFourierNet_one_pipe(torch.nn.Module):
    def __init__(self, sigma, *args, **kwargs):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(input_dimension=4, output_dimension=100, sigma=sigma)
        self.layers = FeedForward(*args, **kwargs)
        self.final_layer = torch.nn.Linear(20, 3)

    def forward(self, x):
        e = self.layers(self.embedding(x))
        out = self.final_layer(e)
        out.labels = ['u', 'y']
        control = LabelTensor(out.extract(['u']) * (1 - x.extract(['x1']) ** 2) * (1 - x.extract(['x2']) ** 2), ['u'])
        state = LabelTensor(out.extract(['y']) * (1 - x.extract(['x1']) ** 2) * (1 - x.extract(['x2']) ** 2), ['y'])
        adjoint = LabelTensor((control.extract(['u']) * x.extract(['mu2'])), ['z'])
        strong_out = control.append(state).append(adjoint)

        return strong_out


class MultiscaleFourierNet_double_pipe(torch.nn.Module):
    def __init__(self, sigma1, sigma2, *args, **kwargs):
        super().__init__()
        self.embedding1 = FourierFeatureEmbedding(input_dimension=4, output_dimension=50, sigma=sigma1)
        self.embedding2 = FourierFeatureEmbedding(input_dimension=4, output_dimension=50, sigma=sigma2)
        self.layers = FeedForward(*args, **kwargs)
        self.final_layer = torch.nn.Linear(20, 3)

    def forward(self, x):
        e1 = self.layers(self.embedding1(x))
        e2 = self.layers(self.embedding2(x))
        out = self.final_layer(torch.cat([e1, e2], dim=-1))
        out.labels = ['u', 'y']
        control = LabelTensor(out.extract(['u']) * (1 - x.extract(['x1']) ** 2) * (1 - x.extract(['x2']) ** 2), ['u'])
        state = LabelTensor(out.extract(['y']) * (1 - x.extract(['x1']) ** 2) * (1 - x.extract(['x2']) ** 2), ['y'])
        adjoint = LabelTensor((control.extract(['u']) * x.extract(['mu2'])), ['z'])
        strong_out = control.append(state).append(adjoint)

        return strong_out