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

area = 60 * np.ones(638)

#Funzione per fare lo scatter plot
class Scatter_plot:
    def plot(self, val, str):
        scatter = plt.scatter(xnp, ynp, s=area, c=val, alpha=1)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Valore')
        plt.savefig(str+'.png')
        plt.close()

# EXTRA FEATURE
class myFeature(torch.nn.Module):

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (-x.extract(['x1']) ** 2 + 1) * (-x.extract(['x2']) ** 2 + 1)
        return LabelTensor(t, ['k0'])

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

    parser = argparse.ArgumentParser(description="Scatter plot")
    parser.add_argument("--checkp", help="directory of the checkpoint to be used", type=str,
                        default="/scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/piarch.ckpt")
    parser.add_argument("--mu1", help="parameter for plotting", type=float)
    parser.add_argument("--mu2", help="parameter for plotting", type=float)
    parser.add_argument("--errors", help="compute the errors", type=int, default=0)
    parser.add_argument("--fem", help="whether or noot to plot fem solutions", type=int, default=0)
    args = parser.parse_args()
    
    #Generating the problem plus discretization
    opc = ParametricEllipticOptimalControl()
    opc.discretise_domain(n=900, mode='random', variables=['x1', 'x2'], locations=['D'])
    opc.discretise_domain(n=50, mode='random', variables=['mu1', 'mu2'], locations=['D'])
    opc.discretise_domain(n=200, mode='random', variables=['x1', 'x2'],
                          locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    opc.discretise_domain(n=50, mode='random', variables=['mu1', 'mu2'],
                          locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])

    model = CustomMultiDFF(
        input_dimensions=5,
        output_dimensions=2,
        layers=[40, 40, 20],
        func=Softplus
    )
    
    #Extra feature
    feat = [myFeature()]
    pinn = PINN.load_from_checkpoint(checkpoint_path=args.checkp, problem=opc, model=model, extra_features=feat)


#Stampo le soluzioni solo se metto in input sia mu1 che mu2 nel parser
if (args.mu1 and args.mu2):
    plotter = Plotter()
    plotter.plot(pinn, fixed_variables={'mu1': args.mu1, 'mu2': args.mu2}, components='u', filename='u_'+str(args.mu1)+'_'+str(args.mu2)+'.png')
    plotter.plot(pinn, fixed_variables={'mu1': args.mu1, 'mu2': args.mu2}, components='z', filename='z_'+str(args.mu1)+'_'+str(args.mu2)+'.png')
    plotter.plot(pinn, fixed_variables={'mu1': args.mu1, 'mu2': args.mu2}, components='y', filename='y_'+str(args.mu1)+'_'+str(args.mu2)+'.png')



##############################################
##############################################
##############################################


#QUI IMPORTO TUTTE LE COSE FEM CON NUMPY
n = 638
path = "/scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/FEM_Elliptic"

# Qua calcoliamo la norma l2 per la soluzione con mu2 = 1 più il grafico
fem_unp = np.load(path + "/alpha_1/control.npy")
fem_ynp = np.load(path + "/alpha_1/state.npy")
fem_znp = np.load(path + "/alpha_1/adjoint.npy")
xnp = np.load(path + "/alpha_1/x.npy")
ynp = np.load(path + "/alpha_1/y.npy")


if args.fem:
    sp = Scatter_plot()
    sp.plot(fem_unp, "fem_u1")
    sp.plot(fem_ynp, "fem_y1")
    sp.plot(fem_znp, "fem_z1")


if args.errors:
    #Qua calcoliamo la norma l2 per la soluzione con mu2 = 1 più il grafico
    fem_u = torch.tensor(fem_unp, dtype=torch.float).view(-1, 1)
    fem_z = torch.tensor(fem_znp, dtype=torch.float).view(-1, 1)
    fem_y = torch.tensor(fem_ynp, dtype=torch.float).view(-1, 1)
    X = torch.tensor(xnp, dtype=torch.float).view(-1, 1)
    Y = torch.tensor(ynp, dtype=torch.float).view(-1, 1)

    x_labelT = LabelTensor(X, ['x1'])
    y_labelT = LabelTensor(Y, ['x2'])
    mu1_labelT = LabelTensor(torch.full((n,), 3, dtype=torch.float).view(-1, 1), ['mu1'])
    mu2_labelT = LabelTensor(torch.full((n,), 1, dtype=torch.float).view(-1, 1), ['mu2'])
    input = x_labelT.append(y_labelT).append(mu1_labelT).append(mu2_labelT)

    output = pinn.forward(input.cuda())

    #Qui bisogna definirli come tensori perché altrimenti non riesce a trasferire su cpu
    errore_u = LabelTensor((output.extract(['u']) - fem_u.cuda()).reshape(n,), ['errore_u'])
    errore_y = LabelTensor((output.extract(['y']) - fem_y.cuda()).reshape(n,), ['errore_y'])
    errore_z = LabelTensor((output.extract(['z']) - fem_z.cuda()).reshape(n,), ['errore_z'])

    #ISSUE : DIVISIONE PER ZERO, COME RISOLVO?
    err_rel_u = abs(errore_u.cpu().detach().numpy())/(fem_unp)
    err_rel_y = abs(errore_y.cpu().detach().numpy())/(fem_ynp)
    err_rel_z = abs(errore_z.cpu().detach().numpy())/(fem_znp)

    #####################
    #Grafici degli errori relativi calcolati rispetto alla soluzione fem
    sp = Scatter_plot()
    sp.plot(err_rel_u, "errore_rel_u1")
    sp.plot(err_rel_y, "errore_rel_y1")
    sp.plot(err_rel_z, "errore_rel_z1")

    ############################################
    #Calcolo errore in norma l2#################
    ############################################

    norma_errore_u = np.linalg.norm(errore_u.cpu().detach().numpy())/np.linalg.norm(fem_u.detach().numpy())
    norma_errore_y = np.linalg.norm(errore_y.cpu().detach().numpy())/np.linalg.norm(fem_y.detach().numpy())
    norma_errore_z = np.linalg.norm(errore_z.cpu().detach().numpy())/np.linalg.norm(fem_z.detach().numpy())

    with open('l2_errors1.txt', 'w') as file:
        file.write(f"norma_errore_u = {norma_errore_u}\n")
        file.write(f"norma_errore_y = {norma_errore_y}\n")
        file.write(f"norma_errore_z = {norma_errore_z}\n")

    ################################################################
    ###############################################################
    ################################################################

# Qua calcoliamo la norma l2 per la soluzione con mu2 = 0.01 più il grafico
fem_unp = np.load(path + "/alpha_0,01/control.npy")
fem_ynp = np.load(path + "/alpha_0,01/state.npy")
fem_znp = np.load(path + "/alpha_0,01/adjoint.npy")
xnp = np.load(path + "/alpha_0,01/x.npy")
ynp = np.load(path + "/alpha_0,01/y.npy")


if args.fem:
    sp = Scatter_plot()
    sp.plot(fem_unp, "fem_u2")
    sp.plot(fem_ynp, "fem_y2")
    sp.plot(fem_znp, "fem_z2")


if args.errors:

    #Qua calcoliamo la norma l2 per la soluzione con mu2 = 0.01 più il grafico
    fem_u = torch.tensor(fem_unp, dtype=torch.float).view(-1, 1)
    fem_z = torch.tensor(fem_znp, dtype=torch.float).view(-1, 1)
    fem_y = torch.tensor(fem_ynp, dtype=torch.float).view(-1, 1)
    X = torch.tensor(xnp, dtype=torch.float).view(-1, 1)
    Y = torch.tensor(ynp, dtype=torch.float).view(-1, 1)

    x_labelT = LabelTensor(X, ['x1'])
    y_labelT = LabelTensor(Y, ['x2'])
    mu1_labelT = LabelTensor(torch.full((n,), 3, dtype=torch.float).view(-1, 1), ['mu1'])
    mu2_labelT = LabelTensor(torch.full((n,), 0.01, dtype=torch.float).view(-1, 1), ['mu2'])
    input = x_labelT.append(y_labelT).append(mu1_labelT).append(mu2_labelT)

    output = pinn.forward(input.cuda())

    #Qui bisogna definirli come tensori perché altrimenti non riesce a trasferire su cpu
    errore_u = LabelTensor((output.extract(['u']) - fem_u.cuda()).reshape(n,), ['errore_u'])
    errore_y = LabelTensor((output.extract(['y']) - fem_y.cuda()).reshape(n,), ['errore_y'])
    errore_z = LabelTensor((output.extract(['z']) - fem_z.cuda()).reshape(n,), ['errore_z'])

    #ISSUE : DIVISIONE PER ZERO, COME RISOLVO?
    err_rel_u = abs(errore_u.cpu().detach().numpy())/(fem_unp)
    err_rel_y = abs(errore_y.cpu().detach().numpy())/(fem_ynp)
    err_rel_z = abs(errore_z.cpu().detach().numpy())/(fem_znp)

    #####################
    #Grafici#############

    #Grafici degli errori relativi calcolati rispetto alla soluzione fem
    sp = Scatter_plot()
    sp.plot(err_rel_u, "errore_rel_u2")
    sp.plot(err_rel_y, "errore_rel_y2")
    sp.plot(err_rel_z, "errore_rel_z2")

    ############################################
    ############################################

    norma_errore_u = np.linalg.norm(errore_u.cpu().detach().numpy())/np.linalg.norm(fem_u.detach().numpy())
    norma_errore_y = np.linalg.norm(errore_y.cpu().detach().numpy())/np.linalg.norm(fem_y.detach().numpy())
    norma_errore_z = np.linalg.norm(errore_z.cpu().detach().numpy())/np.linalg.norm(fem_z.detach().numpy())

    with open('l2_errors2.txt', 'w') as file:
        file.write(f"norma_errore_u = {norma_errore_u}\n")
        file.write(f"norma_errore_y = {norma_errore_y}\n")
        file.write(f"norma_errore_z = {norma_errore_z}\n")
