## PI-ARCH

# import sys
# sys.path.append('C:/Users/Andrea/Desktop/Poli/Tesi magistrale/reporitory_SISSA_PoliTO')

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import Softplus, Tanh

from pina import LabelTensor
from pina.solvers import PINN, NTKPINN
from pina.model import MultiFeedForward
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callbacks import MetricTracker

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

class CustomMultiDFF_strong(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = FeedForward(*args, **kwargs)

    def forward(self, x):
        out = self.model(x)
        out.labels = ['u', 'y']
        control = LabelTensor(out.extract(['u']) * (1 - x.extract(['x1']) ** 2)*(1 - x.extract(['x2']) ** 2), ['u'])
        state = LabelTensor(out.extract(['y']) * (1 - x.extract(['x1']) ** 2)*(1 - x.extract(['x2']) ** 2), ['y'])
        adjoint = LabelTensor((control.extract(['u']) * x.extract(['mu2'])), ['z'])
        strong_out = control.append(state).append(adjoint)
        return strong_out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--strong", help="Whether or not to pose the strong dirichlet", type=int, default=0)
    parser.add_argument("--features", help="extra features", type=int, default=1)
    parser.add_argument("--epochs", help="number of epochs to be trained", type=int, default=50000)
    parser.add_argument("--lr", help="learning rate used for training", type=float, default=0.0005)
    parser.add_argument("--seed", help="seed of the simulation, it gives also the name to all the output in order to distinguish", type=int, default=1000)
    parser.add_argument("--physic_sampling", help="sampling technique for the discretization of the physical domain", type=str, default="grid")
    parser.add_argument("--parametric_sampling", help="sampling technique for the discretization of the parametric domain", type=str, default="grid")
    parser.add_argument("--func", help="activation function", type=str, default="tanh")
    parser.add_argument("--ntk", help="Use the adaptive ntk-based weights", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)

    # extra features
    feat = [myFeature()] if args.features else []
    args = parser.parse_args()

    opc = ParametricEllipticOptimalControl()
    # Create problem and discretise domain
    # Se è grid bisogna usare una discretizzazione diversa
    #physical domain
    if (args.physic_sampling == 'grid' or args.physic_sampling == 'chebyshev'):
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(900)), mode=args.physic_sampling, variables=['x1', 'x2'], locations=['state_eq'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(900)), mode=args.physic_sampling, variables=['x1', 'x2'], locations=['adjoint_eq'])
    else:
        torch.manual_seed(seed); opc.discretise_domain(n=900, mode=args.physic_sampling, variables=['x1', 'x2'], locations=['adjoint_eq'])
        torch.manual_seed(seed); opc.discretise_domain(n=900, mode=args.physic_sampling, variables=['x1', 'x2'], locations=['state_eq'])
    #parametic domain
    if (args.parametric_sampling == "grid" or args.parametric_sampling == 'chebyshev'):
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(50)), mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['state_eq'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(50)), mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['adjoint_eq'])
    else:
        torch.manual_seed(seed); opc.discretise_domain(n=50, mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['state_eq'])
        torch.manual_seed(seed); opc.discretise_domain(n=50, mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['adjoint_eq'])

    if args.func == "tanh":
        func = Tanh
    elif args.func == "softplus":
        func = Softplus

    if (args.strong):
        model = CustomMultiDFF_strong(input_dimensions=4+len(feat),output_dimensions=2,layers=[40, 40, 20],func=func)
    else:
        model = CustomMultiDFF(input_dimensions=4+len(feat),output_dimensions=2,layers=[40, 40, 20],func=func)

    # Creazione dell''stanza di PINN
    if args.ntk == 1:
        pinn = NTKPINN(problem=opc, model=model, optimizer_kwargs={'lr': args.lr}, extra_features=feat)
    else:
        pinn = PINN(problem=opc, model=model, optimizer_kwargs={'lr': args.lr}, extra_features=feat)
    # Creazione di istanza di Trainer
    directory = 'pina.parametric_optimal_control_{}'.format(bool(args.features))

    track = MetricTracker() #Questo serve per tenere conto di tutte le loss

    trainer = Trainer(solver=pinn, accelerator='gpu', max_epochs=args.epochs, callbacks=[track])
    trainer.train()   # Training

    plotter = Plotter()
    #plotter.plot(solver=pinn, filename=args.name + ".png")
    plotter.plot_loss(trainer=trainer, filename=args.name + '_loss' + '.png', metrics=['state_eq_loss', 'adjoint_eq_loss'], logx=True, logy=True)

    # Estrazione di tutte le loss
    # D_loss = track.metrics['D_loss'].cpu().numpy().reshape((args.epochs,1))
    # All_loss = np.concatenate((D_loss), axis = 1)
    # np.save("All_loss.npy", All_loss)

