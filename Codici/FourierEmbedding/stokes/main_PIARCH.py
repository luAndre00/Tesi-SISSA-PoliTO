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
from modelli import MultiscaleFourierNet_one_pipe, MultiscaleFourierNet_double_pipe

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--strong", help="Whether or not to pose the strong dirichlet", type=int, default=1)
    parser.add_argument("--epochs", help="number of epochs to be trained", type=int, default=50000)
    parser.add_argument("--lr", help="learning rate used for training", type=float, default=0.0005)
    parser.add_argument("--seed", help="seed of the simulation, it gives also the name to all the output in order to distinguish", type=int, default=1000)
    parser.add_argument("--physic_sampling", help="sampling technique for the discretization of the physical domain", type=str, default="latin")
    parser.add_argument("--parametric_sampling", help="sampling technique for the discretization of the parametric domain", type=str, default="latin")
    parser.add_argument("--neurons", help="number of neurons of the net", type=int, default=100)
    parser.add_argument("--npname", help="name of the np file", type=str, default="All_loss")
    parser.add_argument("--sigma1", help="first sigma", type=float)
    parser.add_argument("--sigma2", help="second sigma", type=float)
    parser.add_argument("--pipes", help="n of pipes of FFE, 1 or 2", type=int)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)

    opc = ParametricStokesOptimalControl()

    #Discretise domain
    #physical domain
    #I punti su tutti i bordi erano 400/1800
    if (args.physic_sampling == 'grid' or args.physic_sampling == 'chebyshev'):
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(400)), mode=args.physic_sampling, variables=['x', 'y'], locations=['D'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(1800)), mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_right', ])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(1800)), mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_left', ])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(1800)), mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_below', ])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(1800)), mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_above', ])
    else:
        torch.manual_seed(seed); opc.discretise_domain(n=400, mode=args.physic_sampling, variables=['x', 'y'], locations=['D'])
        torch.manual_seed(seed); opc.discretise_domain(n=1800, mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_right'])
        torch.manual_seed(seed); opc.discretise_domain(n=1800, mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_above'])
        torch.manual_seed(seed); opc.discretise_domain(n=1800, mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_below'])
        torch.manual_seed(seed); opc.discretise_domain(n=1800, mode=args.physic_sampling, variables=['x', 'y'], locations=['gamma_left'])
    #parametric domain
    if (args.parametric_sampling == 'grid' or args.parametric_sampling == 'chebyshev'):
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(10)), mode=args.parametric_sampling, variables=['mu'], locations=['D'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(10)), mode=args.parametric_sampling, variables=['mu'], locations=['gamma_right'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(10)), mode=args.parametric_sampling, variables=['mu'], locations=['gamma_above'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(10)), mode=args.parametric_sampling, variables=['mu'], locations=['gamma_left'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(10)), mode=args.parametric_sampling, variables=['mu'], locations=['gamma_below'])
    else:
        torch.manual_seed(seed); opc.discretise_domain(n=10, mode=args.parametric_sampling, variables=['mu'], locations=['D'])
        torch.manual_seed(seed); opc.discretise_domain(n=10, mode=args.parametric_sampling, variables=['mu'], locations=['gamma_right'])
        torch.manual_seed(seed); opc.discretise_domain(n=10, mode=args.parametric_sampling, variables=['mu'], locations=['gamma_left'])
        torch.manual_seed(seed); opc.discretise_domain(n=10, mode=args.parametric_sampling, variables=['mu'], locations=['gamma_above'])
        torch.manual_seed(seed); opc.discretise_domain(n=10, mode=args.parametric_sampling, variables=['mu'], locations=['gamma_below'])

    half_neurons = int(args.neurons/2)

    if args.pipes == 1:
        model = MultiscaleFourierNet_one_pipe(func = torch.nn.Tanh, sigma = args.sigma1, neurons = args.neurons,
                                               layers = [args.neurons, args.neurons, args.neurons, args.neurons],
                                              input_dimensions = args.neurons, output_dimensions = args.neurons)
    elif args.pipes == 2:
        model = MultiscaleFourierNet_double_pipe(input_dimensions = half_neurons, output_dimensions = half_neurons,
                                            layers = [half_neurons, half_neurons, half_neurons, half_neurons],
                                                 func = torch.nn.Tanh, sigma1 = args.sigma1, sigma2 = args.sigma2, neurons = args.neurons)


    pinn = PINN(problem=opc, model=model, optimizer_kwargs={'lr': args.lr})
    
    #Training
    track = MetricTracker()
    trainer = Trainer(solver=pinn, accelerator='gpu', max_epochs=args.epochs, callbacks=[track])
    trainer.train() 

    # Estrazione di tutte le loss 
    gamma1_loss = track.metrics['gamma_above_loss'].cpu().numpy().reshape((args.epochs, 1))
    gamma2_loss = track.metrics['gamma_below_loss'].cpu().numpy().reshape((args.epochs, 1))
    gamma3_loss = track.metrics['gamma_left_loss'].cpu().numpy().reshape((args.epochs, 1))
    gamma4_loss = track.metrics['gamma_right_loss'].cpu().numpy().reshape((args.epochs, 1))
    D_loss = track.metrics['D_loss'].cpu().numpy().reshape((args.epochs, 1))
    mean_loss = (gamma1_loss + gamma2_loss + gamma3_loss + gamma4_loss + D_loss) / 5
    All_loss = np.concatenate((gamma1_loss, gamma2_loss, gamma3_loss, gamma4_loss, D_loss, mean_loss), axis=1)
    np.save(args.npname + ".npy", All_loss)



