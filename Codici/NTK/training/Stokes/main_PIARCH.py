## PI-ARCH
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import Softplus, Tanh
from pina.solvers import PINN, NTKPINN
from pina.trainer import Trainer
from pina.callbacks import MetricTracker
from problem_PIARCH import ParametricStokesOptimalControl
from pina.plotter import Plotter
from modelli import PIARCH_net, PIARCH_net_strong, MultiscaleFourierNet, MultiscaleFourierNet_strong

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--func", help="activation function", type=str, default="tanh")
    parser.add_argument("--strong", help="Whether or not to pose the strong dirichlet", type=int, default=0)
    parser.add_argument("--epochs", help="number of epochs to be trained", type=int, default=50000)
    parser.add_argument("--lr", help="learning rate used for training", type=float, default=0.0005)
    parser.add_argument("--seed", help="seed of the simulation, it gives also the name to all the output in order to distinguish", type=int, default=1000)
    parser.add_argument("--physic_sampling", help="sampling technique for the discretization of the physical domain", type=str, default="latin")
    parser.add_argument("--parametric_sampling", help="sampling technique for the discretization of the parametric domain", type=str, default="latin")
    parser.add_argument("--neurons", help="number of neurons of the net", type=int, default=100)
    parser.add_argument("--fourier", help="boolean var, state if you want fourier net", type=int, default=0)
    parser.add_argument("--npname", help="name of the np file", type=str, default="All_loss")
    parser.add_argument("--ntk", help="Use the adaptive ntk-based weights", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)

    opc = ParametricStokesOptimalControl()

    #Discretise domain
    #physical domain
    #I punti su tutti i bordi erano 400/1800
    if (args.physic_sampling == 'grid' or args.physic_sampling == 'chebyshev'):
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(400)), mode=args.physic_sampling, variables=['x', 'y'], locations=['state'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(400)), mode=args.physic_sampling, variables=['x', 'y'], locations=['adjoint'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(1800)), mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_right'])
    else:
        torch.manual_seed(seed); opc.discretise_domain(n=400, mode=args.physic_sampling, variables=['x', 'y'], locations=['state'])
        torch.manual_seed(seed); opc.discretise_domain(n=400, mode=args.physic_sampling, variables=['x', 'y'], locations=['adjoint'])
        torch.manual_seed(seed); opc.discretise_domain(n=1800, mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_right'])
    #parametric domain
    if (args.parametric_sampling == 'grid' or args.parametric_sampling == 'chebyshev'):
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(10)), mode=args.parametric_sampling, variables=['mu'], locations=['state'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(10)), mode=args.parametric_sampling, variables=['mu'], locations=['adjoint'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(10)), mode=args.parametric_sampling, variables=['mu'], locations=['gamma_right'])
    else:
        torch.manual_seed(seed); opc.discretise_domain(n=10, mode=args.parametric_sampling, variables=['mu'], locations=['state'])
        torch.manual_seed(seed); opc.discretise_domain(n=10, mode=args.parametric_sampling, variables=['mu'], locations=['adjoint'])
        torch.manual_seed(seed); opc.discretise_domain(n=10, mode=args.parametric_sampling, variables=['mu'], locations=['gamma_right'])

    if args.func == "softplus":
        func = Softplus
    elif args.func == "tanh":
        func = Tanh

    if (args.strong == 1 and args.fourier == 1):
        model = MultiscaleFourierNet_strong(input_dimension = 100, output_dimension = 100, 
                                            layers = [args.neurons, args.neurons, args.neurons, args.neurons,], func = func)
    elif (args.strong == 1 and args.fourier == 0):
        model = PIARCH_net_strong(input_dimensions=3, output_dimensions=6, 
                                  layers=[args.neurons, args.neurons, args.neurons, args.neurons], func=func)
    elif (args.strong == 0 and args.fourier == 1):
        model = MultiscaleFourierNet(input_dimensions=100, output_dimensions=100, 
                                  layers=[args.neurons, args.neurons, args.neurons, args.neurons], func=func) 
    else:
        model = PIARCH_net(input_dimensions=3, output_dimensions=6, 
                                  layers=[args.neurons, args.neurons, args.neurons, args.neurons], func=func) 
    
    # Creazione dell''stanza di PINN
    if args.ntk == 1:
        pinn = NTKPINN(problem=opc, model=model, optimizer_kwargs={'lr': args.lr})
    else:
        pinn = PINN(problem=opc, model=model, optimizer_kwargs={'lr': args.lr})
    
    #Training
    track = MetricTracker()
    trainer = Trainer(solver=pinn, accelerator='gpu', max_epochs=args.epochs, callbacks=[track])
    trainer.train() 

    # Estrazione di tutte le loss
    plotter = Plotter()
    plotter.plot(solver=pinn, filename=args.name + ".png")
    plotter.plot_loss(trainer=trainer, filename=args.name + '_loss' + '.png', metrics=['state_loss', 'adjoint_loss', 'gamma_right_loss'], logx=True, logy=True)

    # gamma4_loss = track.metrics['gamma_right_loss'].cpu().numpy().reshape((args.epochs, 1))
    # D_loss = track.metrics['D_loss'].cpu().numpy().reshape((args.epochs, 1))
    # mean_loss = (gamma1_loss + gamma2_loss + gamma3_loss + gamma4_loss + D_loss) / 5
    # All_loss = np.concatenate((gamma1_loss, gamma2_loss, gamma3_loss, gamma4_loss, D_loss, mean_loss), axis=1)
    # np.save(args.npname + ".npy", All_loss)


