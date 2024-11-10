import sys
sys.path.append('scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic')

import argparse
import torch
from problem_PIARCH import ParametricEllipticOptimalControl
from pina.plotter import Plotter
from torch.nn import Softplus
from pina.model import FeedForward
from pina.solvers import PINN
from pina import LabelTensor
import numpy as np
import matplotlib.pyplot as plt
from ex_classes import plot_loss
from ex_classes import Scatter_plot

# EXTRA FEATURE
class myFeature(torch.nn.Module):

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (-x.extract(['x1']) ** 2 + 1) * (-x.extract(['x2']) ** 2 + 1)
        return LabelTensor(t, ['k0'])


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

    path_load = "/scratch/atataran/Tesi-SISSA-PoliTO/Codici/NTK/extraction/poisson_extraction/checkpoint"
    root_err = "/scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/FEM_Elliptic"
    parser = argparse.ArgumentParser(description="Information extraction")
    parser.add_argument("--strong", help="Whether or not to pose the strong dirichlet", type=int, default=0)
    parser.add_argument("--extra", help = "whether or not to use the extra feature", type=int, default=1)
    parser.add_argument("--checkp", help="directory of the checkpoint to be used", type=str, 
                        default="baseline")
    parser.add_argument("--mu1", help="parameter for plotting and for computing the error", type=float)
    parser.add_argument("--mu2", help="parameter for plotting and for computing the error", type=float)
    parser.add_argument("--errors", help="compute the errors", type=int, default=1)
    parser.add_argument("--fem", help="whether or not to plot the fem solutoins", type=int, default=1)
    parser.add_argument("--path_err", help="the path of the fem solutions used to compute the errors", type=str)
    parser.add_argument("--samples", help="If yes, then all the error plots will be visualized with the sample points the PINN has been trained on", type=int, default=1)
    parser.add_argument("--physic_sampling", help="sampling technique for the discretization of the physical domain", type=str, default="grid")
    parser.add_argument("--parametric_sampling", help="sampling technique for the discretization of the parametric domain", type=str, default="grid")
    args = parser.parse_args()

    seed = 1000
    torch.manual_seed(seed)

    # Generating the problem plus discretization
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

    sample_points = opc.input_pts

    #Instance for extra feature
    feat = [myFeature()] if args.extra else []

    if (args.strong):
        model = CustomMultiDFF_strong(input_dimensions=4+len(feat),output_dimensions=2,layers=[40, 40, 20],func=torch.nn.Tanh)
    else:
        model = CustomMultiDFF(input_dimensions=4+len(feat),output_dimensions=2,layers=[40, 40, 20],func=torch.nn.Tanh)
        
    #Extra feature
    checkpoint_completo = path_load + "/ntk.ckpt"
    pinn = PINN.load_from_checkpoint(checkpoint_path=checkpoint_completo, problem=opc, model=model, extra_features=feat)

#Stampo le soluzioni solo se metto in input sia mu1 che mu2 nel parser
if (args.mu1 and args.mu2):
    plotter = Plotter()
    plotter.plot(pinn, fixed_variables={'mu1': args.mu1, 'mu2': args.mu2}, components='u', filename='u_'+str(args.mu1)+'_'+str(args.mu2)+'.png', method = "pcolor")
    plotter.plot(pinn, fixed_variables={'mu1': args.mu1, 'mu2': args.mu2}, components='z', filename='z_'+str(args.mu1)+'_'+str(args.mu2)+'.png', method = "pcolor")
    plotter.plot(pinn, fixed_variables={'mu1': args.mu1, 'mu2': args.mu2}, components='y', filename='y_'+str(args.mu1)+'_'+str(args.mu2)+'.png', method = "pcolor")

##############################################
##############################################
##############################################


#QUI IMPORTO TUTTE LE COSE FEM CON NUMPY
n = 638

#Qua scrivo i suffissi perché altrimenti li devo riscrivere 100 volte
suf_u = "_u_" + str(args.mu1) + "_" + str(args.mu2)
suf_y = "_y_" + str(args.mu1) + "_" + str(args.mu2)
suf_z = "_z_" + str(args.mu1) + "_" + str(args.mu2)

# Qua calcoliamo la norma l2 per la soluzione con mu2 = 1 più il grafico
fem_unp = np.load(root_err + "/" + args.path_err + "/control.npy")
fem_ynp = np.load(root_err + "/" + args.path_err + "/state.npy")
fem_znp = np.load(root_err + "/" + args.path_err + "/adjoint.npy")
xnp = np.load(root_err + "/" + args.path_err + "/x.npy")
ynp = np.load(root_err + "/" + args.path_err + "/y.npy")

if args.fem:
    sp = Scatter_plot()
    sp.plot(fem_unp, x = xnp, y = ynp, str = "fem" + suf_u)
    sp.plot(fem_ynp, x = xnp, y = ynp, str = "fem" + suf_y)
    sp.plot(fem_znp, x = xnp, y = ynp, str = "fem" + suf_z)

if args.errors:
    #Qua calcoliamo la norma l2 per la soluzione con mu2 = 1 più il grafico
    fem_u = torch.tensor(fem_unp, dtype=torch.float).view(-1, 1)
    fem_z = torch.tensor(fem_znp, dtype=torch.float).view(-1, 1)
    fem_y = torch.tensor(fem_ynp, dtype=torch.float).view(-1, 1)
    X = torch.tensor(xnp, dtype=torch.float).view(-1, 1)
    Y = torch.tensor(ynp, dtype=torch.float).view(-1, 1)

    x_labelT = LabelTensor(X, ['x1'])
    y_labelT = LabelTensor(Y, ['x2'])
    mu1_labelT = LabelTensor(torch.full((n,), args.mu1, dtype=torch.float).view(-1, 1), ['mu1'])
    mu2_labelT = LabelTensor(torch.full((n,), args.mu2, dtype=torch.float).view(-1, 1), ['mu2'])
    input = x_labelT.append(y_labelT).append(mu1_labelT).append(mu2_labelT)

    output = pinn.forward(input.cuda())

    #Qui bisogna definirli come tensori perché altrimenti non riesce a trasferire su cpu
    #Errore assoluto
    errore_u = abs(LabelTensor((output.extract(['u']) - fem_u.cuda()).reshape(n,), ['errore_u']).cpu().detach().numpy())
    errore_y = abs(LabelTensor((output.extract(['y']) - fem_y.cuda()).reshape(n,), ['errore_y']).cpu().detach().numpy())
    errore_z = abs(LabelTensor((output.extract(['z']) - fem_z.cuda()).reshape(n,), ['errore_z']).cpu().detach().numpy())

    #Erroe relativo
    err_rel_u = errore_u/np.linalg.norm(fem_unp)
    err_rel_y = errore_y/np.linalg.norm(fem_ynp)
    err_rel_z = errore_z/np.linalg.norm(fem_znp)

    #####################
    #Grafici degli errori relativi e assoluti calcolati rispetto alla soluzione fem e poi anche i grafici della net, perche quelli fatti
    #da pina non sono buoni
    sp = Scatter_plot()
    if (args.physic_sampling == 'grid' or args.physic_sampling == 'chebyshev'):
        sp.plots_grid(err_rel_u, x = xnp, y = ynp, str = "errore_rel" + suf_u, sample_points = sample_points)
        sp.plots_grid(err_rel_y, x = xnp, y = ynp, str = "errore_rel" + suf_y, sample_points = sample_points)
        sp.plots_grid(err_rel_z, x = xnp, y = ynp, str = "errore_rel" + suf_z, sample_points = sample_points)

        sp.plots_grid(errore_u, x = xnp, y = ynp, str = "errore_abs" + suf_u, sample_points = sample_points)
        sp.plots_grid(errore_y, x = xnp, y = ynp, str = "errore_abs" + suf_y, sample_points = sample_points)
        sp.plots_grid(errore_z, x = xnp, y = ynp, str = "errore_abs" + suf_z, sample_points = sample_points)
    else:
        sp.plots(err_rel_u, x = xnp, y = ynp, str = "errore_rel" + suf_u, sample_points = sample_points)
        sp.plots(err_rel_y, x = xnp, y = ynp, str = "errore_rel" + suf_y, sample_points = sample_points)
        sp.plots(err_rel_z, x = xnp, y = ynp, str = "errore_rel" + suf_z, sample_points = sample_points)

        sp.plots(errore_u, x = xnp, y = ynp, str = "errore_abs" + suf_u, sample_points = sample_points)
        sp.plots(errore_y, x = xnp, y = ynp, str = "errore_abs" + suf_y, sample_points = sample_points)
        sp.plots(errore_z, x = xnp, y = ynp, str = "errore_abs" + suf_z, sample_points = sample_points)
        
    sp.plot(output.extract(['u']).cpu().detach().numpy(), x = xnp, y = ynp, str = 'u_'+str(args.mu1)+'_'+str(args.mu2))
    sp.plot(output.extract(['y']).cpu().detach().numpy(), x = xnp, y = ynp, str = 'y_'+str(args.mu1)+'_'+str(args.mu2))
    sp.plot(output.extract(['z']).cpu().detach().numpy(), x = xnp, y = ynp, str = 'z_'+str(args.mu1)+'_'+str(args.mu2))

    #####################################################################
    #Calcolo errore in norma l2 (norma della differenza)#################
    #####################################################################

    norma_errore_u = np.linalg.norm(errore_u)/np.linalg.norm(fem_unp)
    norma_errore_y = np.linalg.norm(errore_y)/np.linalg.norm(fem_ynp)
    norma_errore_z = np.linalg.norm(errore_z)/np.linalg.norm(fem_znp)

    abs_err_u = np.linalg.norm(errore_u)
    abs_err_y = np.linalg.norm(errore_y)
    abs_err_z = np.linalg.norm(errore_z)

    with open('l2_errors_' + str(args.mu1) + '_' + str(args.mu2) + '.txt', 'w') as file:
        file.write(f"errore relativo_u = {norma_errore_u}\n")
        file.write(f"errore relativo_y = {norma_errore_y}\n")
        file.write(f"errore relativo_z = {norma_errore_z}\n")
        file.write(f"errore assoluto_u = {abs_err_u}\n")
        file.write(f"errore assoluto_y = {abs_err_y}\n")
        file.write(f"errore assoluto_z = {abs_err_z}\n")

#Qui plotto tutte le loss in scala loglog
# Loss = np.load(checkpoint_completo + "_piarch.npy")
# pl = plot_loss()
# pl.all_plots(Loss)
#
# #Qui plotto i sample points relativi solamente ai parametri
# sp.plot_parameter(sample_points = sample_points, str = "parameters")