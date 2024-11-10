## PI-ARCH
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn import Softplus, Tanh
from pina.solvers import PINN, NTKPINN
from pina.trainer import Trainer
from pina.callbacks import MetricTracker
from problem_PIARCH import ParametricEllipticOptimalControl
from modelli import MultiscaleFourierNet_one_pipe, MultiscaleFourierNet_double_pipe
from pina import LabelTensor

class myFeature(torch.nn.Module):

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (-x.extract(['x1']) ** 2 + 1) * (-x.extract(['x2']) ** 2 + 1)
        return LabelTensor(t, ['k0'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--strong", help="Whether or not to pose the strong dirichlet", type=int, default=1)
    parser.add_argument("--epochs", help="number of epochs to be trained", type=int, default=50000)
    parser.add_argument("--lr", help="learning rate used for training", type=float, default=0.0005)
    parser.add_argument("--seed", help="seed of the simulation, it gives also the name to all the output in order to distinguish", type=int, default=1000)
    parser.add_argument("--physic_sampling", help="sampling technique for the discretization of the physical domain", type=str, default="grid")
    parser.add_argument("--parametric_sampling", help="sampling technique for the discretization of the parametric domain", type=str, default="grid")
    parser.add_argument("--npname", help="name of the np file", type=str, default="All_loss")
    parser.add_argument("--sigma1", help="first sigma", type=float)
    parser.add_argument("--sigma2", help="second sigma", type=float)
    parser.add_argument("--pipes", help="n of pipes of FFE, 1 or 2", type=int)
    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)

    opc = ParametricEllipticOptimalControl()
    if (args.physic_sampling == 'grid' or args.physic_sampling == 'chebyshev'):
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(900)), mode=args.physic_sampling, variables=['x1', 'x2'], locations=['D'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(200)), mode=args.physic_sampling, variables=['x1', 'x2'], locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    else:
        torch.manual_seed(seed); opc.discretise_domain(n=900, mode=args.physic_sampling, variables=['x1', 'x2'], locations=['D'])
        torch.manual_seed(seed); opc.discretise_domain(n=200, mode=args.physic_sampling, variables=['x1', 'x2'], locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    #parametic domain
    if (args.parametric_sampling == "grid" or args.parametric_sampling == 'chebyshev'):
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(50)), mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['D'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(50)), mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['gamma1'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(50)), mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['gamma2'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(50)), mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['gamma3'])
        torch.manual_seed(seed); opc.discretise_domain(n=int(np.sqrt(50)), mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['gamma4'])
    else:
        torch.manual_seed(seed); opc.discretise_domain(n=50, mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['D'])
        torch.manual_seed(seed); opc.discretise_domain(n=50, mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['gamma1'])
        torch.manual_seed(seed); opc.discretise_domain(n=50, mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['gamma2'])
        torch.manual_seed(seed); opc.discretise_domain(n=50, mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['gamma3'])
        torch.manual_seed(seed); opc.discretise_domain(n=50, mode=args.parametric_sampling, variables=['mu1', 'mu2'], locations=['gamma4'])

    if args.pipes == 1:
        model = MultiscaleFourierNet_one_pipe(func = torch.nn.Tanh, sigma = args.sigma1,
                                               layers = [40, 40, 20],
                                              input_dimensions = 100, output_dimensions = 20)
    elif args.pipes == 2:
        model = MultiscaleFourierNet_double_pipe(input_dimensions = 50, output_dimensions = 10,
                                            layers = [20, 20, 10],
                                                 func = torch.nn.Tanh, sigma1 = args.sigma1, sigma2 = args.sigma2)


    pinn = PINN(problem=opc, model=model, optimizer_kwargs={'lr': args.lr}, extra_features = [])
    
    #Training
    track = MetricTracker()
    trainer = Trainer(solver=pinn, accelerator='gpu', max_epochs=args.epochs, callbacks=[track])
    trainer.train() 

    # Estrazione di tutte le loss 
    gamma1_loss = track.metrics['gamma1_loss'].cpu().numpy().reshape((args.epochs, 1))
    gamma2_loss = track.metrics['gamma2_loss'].cpu().numpy().reshape((args.epochs, 1))
    gamma3_loss = track.metrics['gamma3_loss'].cpu().numpy().reshape((args.epochs, 1))
    gamma4_loss = track.metrics['gamma4_loss'].cpu().numpy().reshape((args.epochs, 1))
    D_loss = track.metrics['D_loss'].cpu().numpy().reshape((args.epochs, 1))
    mean_loss = (gamma1_loss + gamma2_loss + gamma3_loss + gamma4_loss + D_loss) / 5
    All_loss = np.concatenate((gamma1_loss, gamma2_loss, gamma3_loss, gamma4_loss, D_loss, mean_loss), axis=1)
    np.save(args.npname + ".npy", All_loss)



