
import torch
import argparse
import numpy as np
from pina.solvers import PINN, NTKPINN
from pina.plotter import Plotter
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callbacks import MetricTracker

from pina import Condition
from pina.geometry import CartesianDomain
from pina.equation import Equation, FixedValue
from pina.problem import SpatialProblem
from pina.operators import laplacian


class Poisson(SpatialProblem):    
    xmin = -np.pi
    xmax = np.pi
    xrange = [xmin, xmax]
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': xrange})
    
    def residual(input_, output_):
        lap = laplacian(output_, input_, components = ['u'], d = ['x'])
        forzante = torch.sin(input_.extract(['x'])) + 0.1 * (alpha ** 2) * torch.sin(alpha*input_.extract(['x']))
        return - lap - forzante

    def truth_solution(self, input_):
        return torch.sin(input_.extract(['x'])) + 0.1 * torch.sin(alpha*input_.extract(['x']))
    
    conditions = {
        'gamma1' : Condition(location = CartesianDomain({'x': xmin}), equation = FixedValue(0, ['u'])),
        'gamma2' : Condition(location = CartesianDomain({'x': xmax}), equation = FixedValue(0, ['u'])),
        'D' : Condition(location = CartesianDomain({'x': xrange}), equation = Equation(residual))
        }

if __name__ == "__main__":

    seed = 1000
    torch.manual_seed(seed)

    problem = Poisson()
    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--epochs", help="extra features", type=int, default=10000)
    parser.add_argument("--alpha", help="frequency of the solution", type=int, default = 150)
    parser.add_argument("--lr", help="learning rate", type=float, default = 0.01) #provare 1e-2
    parser.add_argument("--sigma", help="variance for the Fourier embedding", type=float)
    parser.add_argument("--ntk", help="Whether to use NTKPINN", type=int, default = 0)
    args = parser.parse_args()

    alpha = torch.tensor(args.alpha)
    
    torch.manual_seed(seed); problem.discretise_domain(n=1, mode='random', variables=['x'], locations=['gamma1'])
    torch.manual_seed(seed); problem.discretise_domain(n=1, mode='random', variables=['x'], locations=['gamma2'])
    torch.manual_seed(seed); problem.discretise_domain(n=900, mode='grid', variables=['x'], locations=['D'])
  
    model = FeedForward(layers = [100, 100], func = torch.nn.Softplus, output_dimensions = 1, input_dimensions = 1)
    
    if args.ntk == 1:
        pinn = PINN(problem=problem, model=model, scheduler=torch.optim.lr_scheduler.MultiStepLR, 
                       scheduler_kwargs={'milestones' : [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 'gamma':0.9})
    else:
        pinn = NTKPINN(problem=problem, model=model, scheduler=torch.optim.lr_scheduler.MultiStepLR, 
                       scheduler_kwargs={'milestones' : [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], 'gamma':0.9})

    track = MetricTracker()
    trainer = Trainer(solver=pinn, accelerator='gpu', max_epochs=args.epochs, callbacks = [track])
    trainer.train()
  
    plotter = Plotter()
    if args.ntk == 1:
        plotter.plot(solver = pinn, filename = "ntk" + "_alpha_" + str(alpha.item()) + ".png")
        plotter.plot_loss(trainer = trainer, filename = "ntk" + '_loss_' + str(alpha.item()) + '.png', 
                          metrics = ['D_loss', 'gamma1_loss', 'gamma2_loss'], logx = True, logy = True)
    else:
        plotter.plot(solver = pinn, filename = "_alpha_" + str(alpha.item()) + ".png")
        plotter.plot_loss(trainer = trainer, filename = '_loss_' + str(alpha.item()) + '.png', 
                          metrics = ['D_loss', 'gamma1_loss', 'gamma2_loss'], logx = True, logy = True)
        
##Scelgo un set di punti dove valutare l'errore della rete:
#xx = np.linspace(-np.pi,np.pi)
#y_true = np.sin(xx) + 0.1*np.sin(alpha*xx)
#xx_torch = torch.from_numpy(xx)
#xx_LabT = LabelTensor







