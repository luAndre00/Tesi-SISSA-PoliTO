## PI-ARCH

# import sys
# sys.path.append('C:/Users/Andrea/Desktop/Poli/Tesi magistrale/reporitory_SISSA_PoliTO')

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import Softplus

from pina import LabelTensor
from pina.solvers import PINN
from pina.model import MultiFeedForward
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callbacks import MetricTracker

from pina import Condition
from pina.geometry import CartesianDomain
from pina.equation import SystemEquation, FixedValue
from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import laplacian

from problem_PIARCH import ParametricEllipticOptimalControl

#############################################


# EXTRA FEATURE
class myFeature(torch.nn.Module):

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (-x.extract(['x1']) ** 2 + 1) * (-x.extract(['x2']) ** 2 + 1)
        return LabelTensor(t, ['k0'])


# Nuovo tipo di modello (fatto da Dario)
class CustomMultiDFF(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = FeedForward(*args, **kwargs)

    def forward(self, x):
        out = self.model(x)
        out.labels = ['u', 'y']
        z = LabelTensor((out.extract(['u']) * x.extract(['mu2'])), ['z'])
        out = out.append(z)
        return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--features", help="extra features", type=int, default=1)
    parser.add_argument("--epochs", help="number of epochs to be trained", type=int, default=10000)
    parser.add_argument("--lr", help="learning rate used for training", type=float, default=0.002)
    parser.add_argument("--seed", help="seed of the simulation, it gives also the name to all the output in order to distinguish",
                        type=int, default=1000)
    # parser.add_argument('-f')  # Serve per risolvere l'errore di sotto
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    nome = str(args.seed)  # Nome che voglio dare alla simulazione così da distinguere loss e plot per tutto

    if args.features is None:
        args.features = 0

    # extra features
    feat = [myFeature()] if args.features else []
    args = parser.parse_args()

    # create problem and discretise domain
    opc = ParametricEllipticOptimalControl()
    opc.discretise_domain(n=900, mode='random', variables=['x1', 'x2'], locations=['D'])
    opc.discretise_domain(n=50, mode='random', variables=['mu1', 'mu2'], locations=['D'])
    opc.discretise_domain(n=200, mode='random', variables=['x1', 'x2'],
                          locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    opc.discretise_domain(n=50, mode='random', variables=['mu1', 'mu2'],
                          locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])

    model = CustomMultiDFF(
        input_dimensions=4+len(feat),
        output_dimensions=2,
        layers=[40, 40, 20],
        func=Softplus
    )
    
    # Creazione dell''stanza di PINN
    pinn = PINN(problem=opc, model=model, optimizer_kwargs={'lr': args.lr}, extra_features=feat)
    # Creazione di istanza di Trainer
    directory = 'pina.parametric_optimal_control_{}'.format(bool(args.features))
    trainer = Trainer(solver=pinn, accelerator='gpu', max_epochs=args.epochs)

    # Training
    trainer.train()


################################################
########################################### LOSS
#Qui salvo la loss function
andamento_loss = trainer._model.lossVec
def salva_variabile(file, variabile):
    with open(file, 'w') as f:
        f.write(repr(variabile))

# # Chiama la funzione per salvare la variabile
salva_variabile('loss_'+ nome +'.txt', andamento_loss) #Qui per salvare la loss

# # Grafico loss
plt.loglog(andamento_loss)
plt.gcf().savefig(nome + '_grafico_loss.pdf', format='pdf') # Qui per salvare il grafico della loss
