seed = 316680
nome = str(seed) #Nome che voglio dare alla simulazione così da distinguere loss e plot per tutto

import torch

import sys
# sys.path.append('C:/Users/Andrea/Desktop/Poli/Tesi magistrale/reporitory_SISSA_PoliTO')

from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import laplacian, grad, div
from pina import Condition, LabelTensor
from pina.geometry import CartesianDomain
from pina.equation import SystemEquation, Equation
from pina.model import MultiFeedForward

import argparse
from torch.nn import Softplus

from pina import Plotter, Trainer
from pina.model import FeedForward
from pina.solvers import PINN

"""Stokes Problem """
# ===================================================== #
#             The Stokes class is defined               #
#       inheriting from SpatialProblem. We  denote:     #
#           ux --> field variable velocity along x      #
#           uy --> field variable velocity along y      #
#           p --> field variable pressure               #
#           x,y --> spatial variables                   #
#                                                       #
#           https://arx#iv.org/pdf/2110.13530.pdf       #
# ===================================================== #
alfa = 0.008

class Stokes(SpatialProblem, ParametricProblem):

    # assign output/ spatial variables
    output_variables = ['vx', 'vy', 'p', 'ux', 'uy', 'r', 'zx', 'zy'] 
    #vx, vy is the 2-dim state variable, #ux, uy is the 2-dim control variable. 
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 2
    mumin = 0.5
    mumax = 1.5

    xrange = [xmin, xmax]
    yrange = [ymin, ymax]
    murange = [mumin, mumax]
    
    #r è la variabile aggiunta della pressione, z è la aggiunta del campo di velocità
    spatial_domain = CartesianDomain({'x': xrange, 'y': yrange})
    parameter_domain = CartesianDomain({'mu': murange})




    
    #Prima ci sono tutte le equazioni sulle variabili aggiunte, poi tutte quelle per le variabili non aggiunte
    #Attenzione che la variabile aggiunta z è stata sostituita ovunque con l'equazione del forward, z si ricava poi
    #rispetto al controllo
    # PDE
    def momentum_ad_x(input_, output_):
        delta = laplacian(output_, input_, components = ['ux'], d = ['x', 'y'])
        return -0.1 * alfa * delta + grad(output_, input_, components = ['r'], d = ['x']) - input_.extract(['y']) + output_.extract(['vx'])
            
    def momentum_ad_y(input_, output_):
        delta = laplacian(output_, input_, components = ['uy'], d = ['x', 'y'])
        return -0.1 * alfa * delta + grad(output_, input_, components = ['r'], d = ['y'])
    
    def continuity_ad(input_, output_):
        return grad(output_, input_, components = ['ux'], d = ['x']) + grad(output_, input_, components = ['uy'], d = ['y'])

    # BOUNDARY CONDITIONS on adjuncted variables
    # Dirichlet
    def dirichlet1_ad(input_, output_):
        return output_.extract(['ux'])

    def dirichlet2_ad(input_, output_):
        return output_.extract(['uy'])

    # Neumann
    def neumann1_ad(input_, output_):
        return -output_.extract(['r']) + 0.1 * alfa * grad(output_, input_, components = ['ux'], d = ['x'])
    def neumann2_ad(input_, output_):
        return output_.extract(['uy'])

    ############################################################################

    # Momentum Equations
    def momentum_x(input_, output_):
        delta = laplacian(output_, input_, components = ['vx'], d = ['x'])
        return -0.1 * delta + grad(output_, input_, components = ['p'], d = ['x'])  - output_.extract(['ux'])
 
    def momentum_y(input_, output_):
        delta = laplacian(output_, input_, components = ['vy'], d = ['y'])
        return -0.1 * delta + grad(output_, input_, components = ['p'], d = ['y']) + input_.extract(['mu'])  - output_.extract(['uy'])

    # Continuity equation
    def continuity(input_, output_):
        return grad(output_, input_, components = ['vx'], d = ['x']) + grad(output_, input_, components = ['vy'], d = ['y'])

    # BOUNDARY CONDITIONS on principal variable
    # Dirichlet
    def dirichlet1(input_, output_):
        return output_.extract(['vx']) - input_.extract(['y'])

    def dirichlet2(input_, output_):
        return output_.extract(['vy'])

    # Neumann
    def neumann1(input_, output_):
        return -output_.extract(['p']) + 0.1*grad(output_, input_, components = ['vx'], d = ['x'])

    def neumann2(input_, output_):
        return output_.extract(['vy'])


    
    
    #Problem Statement
    conditions = {
        'gamma_above': Condition(location=CartesianDomain({'x': xrange, 'y':  2, 'mu': murange}), equation=SystemEquation([dirichlet1, dirichlet2, dirichlet1_ad, dirichlet2_ad])), #Dirichlet
        'gamma_left': Condition(location=CartesianDomain({'x': 0, 'y': yrange, 'mu': murange}), equation=SystemEquation([dirichlet1, dirichlet2, dirichlet1_ad, dirichlet2_ad])), #Dirichlet
        'gamma_below': Condition(location=CartesianDomain({'x':  xrange, 'y': 0, 'mu': murange}), equation=SystemEquation([dirichlet1, dirichlet2, dirichlet1_ad, dirichlet2_ad])), #Dirichlet
        'gamma_right':  Condition(location=CartesianDomain({'x': 1, 'y': yrange, 'mu': murange}), equation=SystemEquation([neumann1, neumann2, neumann1_ad, neumann2_ad])), #Neumann
        'D': Condition(location=CartesianDomain({'x': xrange, 'y': yrange, 'mu': murange}), equation=SystemEquation([momentum_x,momentum_y,continuity, 
                                                                                                                     momentum_ad_x,momentum_ad_y,continuity_ad]))
    }

######################################################################

class CustomMultiDFF(MultiFeedForward):

    def __init__(self, dff_dict):
        super().__init__(dff_dict)

    # Questo si usa al posto della equazione che "manca", altrimenti viene male
    def forward(self, x):
        out = self.uu(x)
        out.labels = ['vx', 'vy', 'p', 'ux', 'uy', 'r']
        z = LabelTensor(alfa * out.extract(['ux', 'uy']), ['zx', 'zy'])

        return out.append(z)


if __name__ == "__main__":
    
    torch.manual_seed(seed)
    
    epochs = 10

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--load", help = "directory to save or load file", type = str)
    parser.add_argument("--epochs", help = "extra features", type = int, default = epochs)
    parser.add_argument('-f') #Serve per risolvere l'errore di sotto
    args = parser.parse_args()
    
    # create problem and discretise domain
    stokes_opc = Stokes()
    stokes_opc.discretise_domain(n = 1800, mode = 'lh', variables = ['x', 'y'], locations = ['gamma_above', 'gamma_left', 'gamma_below', 'gamma_right'])
    stokes_opc.discretise_domain(n = 400,  mode = 'lh', variables = ['x', 'y'], locations = ['D'])
    stokes_opc.discretise_domain(n = 10,  mode = 'lh', variables = ['mu'], locations = ['gamma_above', 'gamma_left', 'gamma_below', 'gamma_right'])
    stokes_opc.discretise_domain(n = 10,  mode = 'lh', variables = ['mu'], locations = ['D'])
    
    # make the model
    model = CustomMultiDFF(
        {'uu': {
                'input_dimensions': 3,
                'output_dimensions': 6,
                'layers': [40, 40, 40, 40],
                'func': Softplus, },})
    
    # make the pinn
    pinn = PINN(problem = stokes_opc, model = model, optimizer_kwargs={'lr' : 0.003})
    
    # create trainer
    directory = 'pina.navier_stokes'
    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=args.epochs, default_root_dir=directory)

    #Training
    trainer.train()

##########################################################
################################################## GRAFICI

plotter = Plotter()
plotter.plot(pinn, fixed_variables={'mu': 0}, components='vx', filename = nome + '_vx.pdf')
plotter.plot(pinn, fixed_variables={'mu': 0}, components='vy', filename = nome + '_vy.pdf')
plotter.plot(pinn, fixed_variables={'mu': 0}, components='ux', filename = nome + '_ux.pdf')
plotter.plot(pinn, fixed_variables={'mu': 0}, components='uy', filename = nome + '_uy.pdf')
plotter.plot(pinn, fixed_variables={'mu': 0}, components='p', filename = nome + '_p.pdf')
plotter.plot(pinn, fixed_variables={'mu': 0}, components='r', filename = nome + '_r.pdf')
plotter.plot(pinn, fixed_variables={'mu': 0}, components='zx', filename = nome + '_zx.pdf')
plotter.plot(pinn, fixed_variables={'mu': 0}, components='zy', filename = nome + '_zy.pdf')


#Qui salvo la loss function
andamento_loss = trainer._model.lossVec
def salva_variabile(file, variabile):
    with open(file, 'w') as f:
        f.write(repr(variabile))

# Chiama la funzione per salvare la variabile
salva_variabile('loss_'+ nome +'.txt', andamento_loss)

# Grafico loss
plt.loglog(andamento_loss)
plt.gcf().savefig(nome + 'grafico_loss.pdf', format='pdf') # Qui per salvare il grafico della loss


