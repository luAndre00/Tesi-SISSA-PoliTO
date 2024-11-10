
import torch
import matplotlib.pyplot as plt
from pina import LabelTensor
from pina.model import FeedForward
from pina.model.layers import FourierFeatureEmbedding

alpha = 0.008

# Nuovo tipo di modello (fatto da Dario)
class MultiscaleFourierNet_one_pipe(torch.nn.Module):
    def __init__(self, sigma, neurons, *args, **kwargs):
        super().__init__()
        self.embedding = FourierFeatureEmbedding(input_dimension=3, output_dimension=neurons, sigma=sigma)
        self.layers = FeedForward(*args, **kwargs)
        self.final_layer = torch.nn.Linear(neurons, 6)

    def forward(self, x):
        e = self.layers(self.embedding(x))
        out = self.final_layer(e)
        out.labels = ['vx', 'vy', 'p', 'ux', 'uy', 'r']
        p = LabelTensor(out.extract(['p']), 'p')
        r = LabelTensor(out.extract(['r']), 'r')
        
        vx = LabelTensor(x.extract(['y']) + out.extract(['vx']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['vx'])
        vy = LabelTensor(out.extract(['vy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['vy'])
        ux = LabelTensor(out.extract(['ux']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['ux'])
        uy = LabelTensor(out.extract(['uy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['uy'])
        #Qua le condizioni al bordo ci sono già ereditate da ux uy
        zx = LabelTensor(ux*alpha, 'zx')
        zy = LabelTensor(uy*alpha, 'zy')

        strong_out = vx.append(vy).append(p).append(ux).append(uy).append(r).append(zx).append(zy)
        return strong_out


class MultiscaleFourierNet_double_pipe(torch.nn.Module):
    def __init__(self, sigma1, sigma2, neurons, *args, **kwargs):
        super().__init__()
        self.embedding1 = FourierFeatureEmbedding(input_dimension=3, output_dimension=int(neurons/2), sigma=sigma1)
        self.embedding2 = FourierFeatureEmbedding(input_dimension=3, output_dimension=int(neurons/2), sigma=sigma2)
        self.layers = FeedForward(*args, **kwargs)
        self.final_layer = torch.nn.Linear(neurons, 6)

    def forward(self, x):
        e1 = self.layers(self.embedding1(x))
        e2 = self.layers(self.embedding2(x))
        out = self.final_layer(torch.cat([e1, e2], dim=-1))
        out.labels = ['vx', 'vy', 'p', 'ux', 'uy', 'r']
        p = LabelTensor(out.extract(['p']), 'p')
        r = LabelTensor(out.extract(['r']), 'r')

        vx = LabelTensor(x.extract(['y']) + out.extract(['vx']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['vx'])
        vy = LabelTensor(out.extract(['vy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['vy'])
        ux = LabelTensor(out.extract(['ux']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']), ['ux'])
        uy = LabelTensor(out.extract(['uy']) * (2 - x.extract(['y'])) * x.extract(['x']) * x.extract(['y']) * (1 - x.extract(['x'])), ['uy'])
        # Qua le condizioni al bordo ci sono già ereditate da ux uy
        zx = LabelTensor(ux * alpha, 'zx')
        zy = LabelTensor(uy * alpha, 'zy')

        strong_out = vx.append(vy).append(p).append(ux).append(uy).append(r).append(zx).append(zy)
        return strong_out