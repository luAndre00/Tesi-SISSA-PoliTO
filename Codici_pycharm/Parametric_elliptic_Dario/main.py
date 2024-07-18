seed = 316680
nome = str(seed)  # Nome che voglio dare alla simulazione così da distinguere loss e plot per tutto

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import Softplus

# Questo serve a fare in modo che tutti i problemi abbiano come riferimento la stessa cartella di PINA
# così non si creano versioni diverse o sovrapposizioni
import sys

from pina import LabelTensor
from pina.solvers import PINN
from pina.model import MultiFeedForward
from pina.plotter import Plotter
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callbacks import MetricTracker

from pina import Condition
from pina.geometry import CartesianDomain
from pina.equation import SystemEquation, FixedValue
from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import laplacian


# ===================================================== #
#           u --> field variable                        #
#           p --> field variable                        #
#           z --> field variable                        #
#           x1, x2 --> spatial variables                #
#           mu, mu2 --> problem parameters            #
#                                                       #
#           https://arxiv.org/pdf/2110.13530.pdf        #
# ===================================================== #


class ParametricEllipticOptimalControl(SpatialProblem, ParametricProblem):
    # setting spatial variables ranges
    xmin, xmax, ymin, ymax = -1, 1, -1, 1
    x_range = [xmin, xmax]
    y_range = [ymin, ymax]
    # setting parameters range
    amin, amax = 0.01, 1
    mumin, mumax = 0.5, 3
    mu_range = [mumin, mumax]
    a_range = [amin, amax]
    # setting field variables
    output_variables = ['u', 'y', 'z']
    # setting spatial and parameter domain
    spatial_domain = CartesianDomain({'x1': x_range, 'x2': y_range})
    parameter_domain = CartesianDomain({'mu1': mu_range, 'mu2': a_range})

    # equation terms as in https://arxiv.org/pdf/2110.13530.pdf
    def term1(input_, output_):
        laplace_z = laplacian(output_, input_, components=['z'], d=['x1', 'x2'])
        return output_.extract(['y']) - laplace_z - input_.extract(['mu1'])

    def term2(input_, output_):
        laplace_y = laplacian(output_, input_, components=['y'], d=['x1', 'x2'])
        return - laplace_y - output_.extract(['u'])

    # setting problem condition formulation
    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x1': x_range, 'x2': 1, 'mu1': mu_range, 'mu2': a_range}),
            equation=FixedValue(0, ['z', 'y'])),
        'gamma2': Condition(
            location=CartesianDomain({'x1': x_range, 'x2': -1, 'mu1': mu_range, 'mu2': a_range}),
            equation=FixedValue(0, ['z', 'y'])),
        'gamma3': Condition(
            location=CartesianDomain({'x1': 1, 'x2': y_range, 'mu1': mu_range, 'mu2': a_range}),
            equation=FixedValue(0, ['z', 'y'])),
        'gamma4': Condition(
            location=CartesianDomain({'x1': -1, 'x2': y_range, 'mu1': mu_range, 'mu2': a_range}),
            equation=FixedValue(0, ['z', 'y'])),
        'D': Condition(location=CartesianDomain({'x1': x_range, 'x2': y_range, 'mu1': mu_range, 'mu2': a_range}),
                       equation=SystemEquation([term1, term2], reduction='sum')), #Senza il reduction sum c'è un bias nella loss
    }


#############################################


# EXTRA FEATURE
class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (-x.extract(['x1']) ** 2 + 1) * (-x.extract(['x2']) ** 2 + 1)
        return LabelTensor(t, ['k0'])


# FORWARD
# class CustomMultiDFF(MultiFeedForward):
#
#     def __init__(self, dff_dict):
#         super().__init__(dff_dict)
#
#     # Original forward
#     def forward(self, x):
#         out = self.uu(x)
#         out.labels = ['u', 'y']
#         p = LabelTensor((out.extract(['u']) * x.extract(['mu2'])), ['p'])
#
#         return out.append(p)

# Nuova tipo di modello
class CustomMultiDFF(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = FeedForward(*args, **kwargs)

    def forward(self, x):
        out = self.model(x)
        out.labels = ['u', 'y']
        p = LabelTensor((out.extract(['u']) * x.extract(['mu2'])), ['p'])
        out = out.append(p)
        return out


if __name__ == "__main__":

    torch.manual_seed(seed)

    epochs = 10000
    flag_extra_feature = 1

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--load", help="directory to save or load file", type=str)
    parser.add_argument("--features", help="extra features", type=int, default=flag_extra_feature)
    parser.add_argument("--epochs", help="extra features", type=int, default=epochs)
    parser.add_argument('-f')  # Serve per risolvere l'errore di sotto
    args = parser.parse_args()

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

    # Architettura
    # model = CustomMultiDFF(
    #     {
    #         'uu': {
    #             'input_dimensions': 4 + len(feat),  # due input spaziali più due parametri
    #             'output_dimensions': 2,
    #             'layers': [40, 40, 20],
    #             'func': Softplus,
    #         },
    #     }
    # )

    model = CustomMultiDFF(
        input_dimensions=4 + len(feat),
        output_dimensions=2,
        layers=[40, 40, 20],
        func=Softplus
    )

    # Creazione dell''stanza di PINN
    pinn = PINN(problem=opc, model=model, optimizer_kwargs={'lr': 0.002}, extra_features=feat)

    # Creazione di istanza di Trainer
    directory = 'pina.parametric_optimal_control_{}'.format(bool(args.features))
    trainer = Trainer(solver=pinn, accelerator='gpu', max_epochs=args.epochs)  # callbacks = [MetricTracker()]

    # Training
    trainer.train()

###############################################
####################################### GRAFICI
plotter = Plotter()
plotter.plot(pinn, fixed_variables={'mu1': 3, 'mu2': 1}, components='u', filename=nome + '_u.png')
plotter.plot(pinn, fixed_variables={'mu1': 3, 'mu2': 1}, components='y', filename=nome + '_y.png')
plotter.plot(pinn, fixed_variables={'mu1': 3, 'mu2': 1}, components='z', filename=nome + '_z.png')
# plotter.plot_loss(trainer, ['gamma1_loss', 'gamma3_loss', 'D_loss', 'gamma2_loss', 'gamma4_loss'], filename='pippo.png')



#############################################CALCOLO DELLA NORMA l2 PER TUTTI GLI OUTPUT

n = 638
path = "/home/atataranni/PINA/Codici_riordinati/Parametric_elliptic/FEM_Elliptic"

fem_u = torch.tensor(np.load(path + "/alpha_1/control.npy"), dtype=torch.float).view(-1, 1)
fem_p = torch.tensor(np.load(path + "/alpha_1/adjoint.npy"), dtype=torch.float).view(-1, 1)
fem_y = torch.tensor(np.load(path + "/alpha_1/state.npy"), dtype=torch.float).view(-1, 1)
fem_x = torch.tensor(np.load(path + "/alpha_1/x.npy"), dtype=torch.float).view(-1, 1)
fem_y = torch.tensor(np.load(path + "/alpha_1/y.npy"), dtype=torch.float).view(-1, 1)

x_labelT = LabelTensor(fem_x, ['x1'])
y_labelT = LabelTensor(fem_y, ['x2'])
mu_labelT = LabelTensor(torch.full((n,), 3, dtype=torch.float).view(-1, 1), ['mu'])
alfa_labelT = LabelTensor(torch.full((n,), 1, dtype=torch.float).view(-1, 1), ['alpha'])
input = x_labelT.append(y_labelT).append(mu_labelT).append(alfa_labelT)

output = pinn.forward(input)
output.labels = ['u', 'y', 'p']

errore_u = (output.extract(['u']) - fem_u).reshape(n,)
errore_y = (output.extract(['y']) - fem_y).reshape(n,)
errore_p = (output.extract(['p']) - fem_p).reshape(n,)

norma_errore_u = torch.dot(errore_u, errore_u).item()
norma_errore_y = torch.dot(errore_y, errore_y).item()
norma_errore_p = torch.dot(errore_p, errore_p).item()

with open(nome + 'l2_errors.txt', 'w') as file:
    file.write(f"norma_errore_u = {norma_errore_u}\n")
    file.write(f"norma_errore_y = {norma_errore_y}\n")
    file.write(f"norma_errore_p = {norma_errore_p}\n")

