import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import Softplus

from pina import LabelTensor
from pina.solvers import PINN
from pina.model import MultiFeedForward
from pina.plotter import Plotter
from pina.trainer import Trainer
from problem import (ParametricEllipticOptimalControl)


torch.manual_seed(316680)

#In questo problema l'extra feature viene usata, è un po' strana
class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (-x.extract(['x1'])**2+1) * (-x.extract(['x2'])**2+1)
        # print(t)
        return LabelTensor(t, ['k0'])


class CustomMultiDFF(MultiFeedForward):

    def __init__(self, dff_dict):
        super().__init__(dff_dict)

    #Questo si usa al posto della equazione che "manca", altrimenti viene male
    def forward(self, x):
        out = self.uu(x)
        out.labels = ['u', 'y']

        u = out.extract(['u'])*(1-x.extract(['x1'])**2)*(1-x.extract(['x2'])**2) #Condizioni al bordo forti
        y = out.extract(['y'])*(1-x.extract(['x1'])**2)*(1-x.extract(['x2'])**2) #Condizioni al bordo forti
        t = torch.cat((u,y), dim=1)

        strong_out = LabelTensor(t, ['u','y'])
        p = LabelTensor(
            (strong_out.extract(['u'])), ['p'])

        return strong_out.append(p)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--load", help="directory to save or load file", type=str)
    parser.add_argument("--features", help="extra features", type=int, default = 1)
    parser.add_argument("--epochs", help="extra features", type=int, default=15000)
    parser.add_argument('-f')#Serve per risolvere l'errore di sotto
    args = parser.parse_args()

    if args.features is None:
        args.features = 0

    # extra features
    feat = [myFeature()] if args.features else []
    args = parser.parse_args()

    # create problem and discretise domain, qua in sostanza genero i punti su cui voglio allenare
    opc = ParametricEllipticOptimalControl() #Creo un istanza di modello
    opc.discretise_domain(n=900, mode='random', variables=['x1', 'x2'], locations=['D'])
    opc.discretise_domain(n=50, mode='random', variables=['mu'], locations=['D'])
    opc.discretise_domain(n=200, mode='random', variables=['x1', 'x2'], locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    opc.discretise_domain(n=50, mode='random', variables=['mu'], locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])

    # create model, qui è un po' più complesso e per costruire il modello si appoggia alla classe
    #che ha creato prima CustomMultiDFF
    model = CustomMultiDFF(
        {
            'uu': {
                'input_dimensions': 3 + len(feat), #due più due parametri
                'output_dimensions': 2,
                'layers': [40, 40, 20],
                'func': Softplus,
            },
        }
    )

    # create PINN
    pinn = PINN(problem=opc, model=model, optimizer_kwargs={'lr' : 0.002}, extra_features=feat)

    # create trainer
    directory = 'pina.parametric_optimal_control_{}'.format(bool(args.features))
    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=args.epochs, default_root_dir=directory)

    if args.load:
        pinn = PINN.load_from_checkpoint(checkpoint_path=args.load, problem=opc, model=model, extra_features=feat)
        plotter = Plotter()
        plotter.plot(pinn, fixed_variables={'mu' : 1 , 'alpha' : 0.001}, components='y')
    else:
        trainer.train()

    plotter = Plotter()
    plotter.plot(pinn, fixed_variables={'mu': 3}, components='y')
    plt.gcf().savefig('epoche.pdf', format='pdf')

    #Qui salvo la loss function
    andamento_loss = trainer._model.lossVec
    def salva_variabile(file, variabile):
        with open(file, 'w') as f:
            f.write(repr(variabile))

    # Chiama la funzione per salvare la variabile
    salva_variabile('loss.txt', andamento_loss)

