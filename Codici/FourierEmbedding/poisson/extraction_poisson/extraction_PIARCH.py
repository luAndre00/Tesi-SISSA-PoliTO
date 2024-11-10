## PI-ARCH
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pina.solvers import PINN
from pina.trainer import Trainer
from pina.callbacks import MetricTracker
from modelli import MultiscaleFourierNet_one_pipe, MultiscaleFourierNet_double_pipe
from pina import LabelTensor
from pina.solvers import PINN
from pina.plotter import Plotter
from problem_PIARCH import ParametricEllipticOptimalControl
from ex_classes import plot_loss
from ex_classes import Scatter_plot

class myFeature(torch.nn.Module):

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (-x.extract(['x1']) ** 2 + 1) * (-x.extract(['x2']) ** 2 + 1)
        return LabelTensor(t, ['k0'])


if __name__ == "__main__":

    path_load = "/scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH/"
    root_err = "/scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/FEM_Stokes"
    parser = argparse.ArgumentParser(description="Information extraction")
    parser.add_argument("--strong", help="Whether or not to pose the strong dirichlet", type=int, default=0)
    parser.add_argument("--extra", help="whether or not to use the extra feature", type=int, default=1)
    parser.add_argument("--checkp", help="directory of the checkpoint to be used", type=str, default="baseline")
    parser.add_argument("--mu", help="parameter mu1 for plotting and for computing the error", type=float)
    parser.add_argument("--errors", help="compute the errors", type=int, default=1)
    parser.add_argument("--fem", help="whether or not to plot the fem solutions", type=int, default=1)
    parser.add_argument("--path_err", help="the path of the fem solutions used to compute the errors", type=str)
    parser.add_argument("--samples", help="If yes, then all the error plots will be visualized with the sample points the PINN has been trained on", type=int, default=1)
    parser.add_argument("--physic_sampling", help="sampling technique for the discretization of the physical domain", type=str, default="latin")
    parser.add_argument("--parametric_sampling", help="sampling technique for the discretization of the parametric domain", type=str, default="latin")
    parser.add_argument("--neurons", help="number of neurons of the net", type=int, default=100)
    parser.add_argument("--func", help="activation function", type=str)
    parser.add_argument("--npname", help="name of the np file", type=str, default="All_loss")
    parser.add_argument("--sigma1", help="first sigma", type=float)
    parser.add_argument("--sigma2", help="second sigma", type=float)
    parser.add_argument("--pipes", help="n of pipes of FFE, 1 or 2", type=int)
    args = parser.parse_args()
    seed = 1000
    torch.manual_seed(seed)

    opc = ParametricEllipticOptimalControl()
    if (args.physic_sampling == 'grid' or args.physic_sampling == 'chebyshev'):
        torch.manual_seed(seed);
        opc.discretise_domain(n=int(np.sqrt(900)), mode=args.physic_sampling, variables=['x1', 'x2'], locations=['D'])
        torch.manual_seed(seed);
        opc.discretise_domain(n=int(np.sqrt(200)), mode=args.physic_sampling, variables=['x1', 'x2'],
                              locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    else:
        torch.manual_seed(seed);
        opc.discretise_domain(n=900, mode=args.physic_sampling, variables=['x1', 'x2'], locations=['D'])
        torch.manual_seed(seed);
        opc.discretise_domain(n=200, mode=args.physic_sampling, variables=['x1', 'x2'],
                              locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    # parametic domain
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

    sample_points = opc.input_pts

    if args.pipes == 1:
        model = MultiscaleFourierNet_one_pipe(func = torch.nn.Tanh, sigma = args.sigma1,
                                               layers = [40, 40, 20],
                                              input_dimensions = 100, output_dimensions = 20)
    elif args.pipes == 2:
        model = MultiscaleFourierNet_double_pipe(input_dimensions = 50, output_dimensions = 10,
                                            layers = [20, 20, 10],
                                                 func = torch.nn.Tanh, sigma1 = args.sigma1, sigma2 = args.sigma2)

    checkpoint_completo = path_load + args.checkp
    pinn = PINN.load_from_checkpoint(checkpoint_path=checkpoint_completo + "_piarch.ckpt", problem=opc, model=model)

# Stampo le soluzioni solo se metto in input mu
# if (args.mu):
#    plotter = Plotter()
#    plotter.plot(pinn, fixed_variables={'mu': args.mu}, components='vx', filename='vx_'+str(args.mu)+'.png', method="pcolor")
#    plotter.plot(pinn, fixed_variables={'mu': args.mu}, components='vy', filename='vy_'+str(args.mu)+'.png', method="pcolor")
#    plotter.plot(pinn, fixed_variables={'mu': args.mu}, components='p', filename='p_'+str(args.mu)+'.png', method="pcolor")
#    plotter.plot(pinn, fixed_variables={'mu': args.mu}, components='ux', filename='ux_'+str(args.mu)+'.png', method="pcolor")
#    plotter.plot(pinn, fixed_variables={'mu': args.mu}, components='uy', filename='uy_'+str(args.mu)+'.png', method="pcolor")
#    plotter.plot(pinn, fixed_variables={'mu': args.mu}, components='r', filename='r_'+str(args.mu)+'.png', method="pcolor")
#    plotter.plot(pinn, fixed_variables={'mu': args.mu}, components='zx', filename='zx_'+str(args.mu)+'.png', method="pcolor")
#    plotter.plot(pinn, fixed_variables={'mu': args.mu}, components='zy', filename='zy_'+str(args.mu)+'.png', method="pcolor")

# Qui plotto tutte le loss in scala loglog
Loss = np.load(checkpoint_completo + "_piarch.npy")
pl = plot_loss()
pl.all_plots(Loss)

##############################################
##############################################
##############################################

n = 756

# Qua scrivo i suffissi perché altrimenti li devo riscrivere 100 volte
suf_vx = "_vx_" + str(args.mu)
suf_vy = "_vy_" + str(args.mu)
suf_ux = "_ux_" + str(args.mu)
suf_uy = "_uy_" + str(args.mu)
suf_p = "_p_" + str(args.mu)
suf_r = "_r_" + str(args.mu)
suf_zx = "_zx_" + str(args.mu)
suf_zy = "_zy_" + str(args.mu)

# Qui importo tutti i risultati della fem più i punti in cui la fem stessa è definita
fem_vxnp = np.load(root_err + "/" + args.path_err + "/vx.npy")
fem_vynp = np.load(root_err + "/" + args.path_err + "/vy.npy")
fem_uxnp = np.load(root_err + "/" + args.path_err + "/ux.npy")
fem_uynp = np.load(root_err + "/" + args.path_err + "/uy.npy")
fem_pnp = np.load(root_err + "/" + args.path_err + "/p.npy")
fem_rnp = np.load(root_err + "/" + args.path_err + "/r.npy")
fem_zxnp = np.load(root_err + "/" + args.path_err + "/zx.npy")
fem_zynp = np.load(root_err + "/" + args.path_err + "/zy.npy")

xnp = np.load(root_err + "/" + args.path_err + "/x.npy")
ynp = np.load(root_err + "/" + args.path_err + "/y.npy")

# Importo anche i dati sul modulo
fem_mag_v = np.load(root_err + "/" + args.path_err + "/mag_v.npy")
fem_mag_u = np.load(root_err + "/" + args.path_err + "/mag_u.npy")
fem_mag_z = np.load(root_err + "/" + args.path_err + "/mag_z.npy")

# Se fem=1 allora plotto i risultati della soluzione agli elementi finiti
if args.fem:
    sp = Scatter_plot()
    sp.plot(fem_vxnp, x=xnp, y=ynp, str="fem" + suf_vx)
    sp.plot(fem_vynp, x=xnp, y=ynp, str="fem" + suf_vy)
    sp.plot(fem_uxnp, x=xnp, y=ynp, str="fem" + suf_ux)
    sp.plot(fem_uynp, x=xnp, y=ynp, str="fem" + suf_uy)
    sp.plot(fem_pnp, x=xnp, y=ynp, str="fem" + suf_p)
    sp.plot(fem_rnp, x=xnp, y=ynp, str="fem" + suf_r)
    sp.plot(fem_zxnp, x=xnp, y=ynp, str="fem" + suf_zx)
    sp.plot(fem_zynp, x=xnp, y=ynp, str="fem" + suf_zy)
    sp.plot(fem_mag_v, x=xnp, y=ynp, str="fem" + "_mag_v_" + str(args.mu))
    sp.plot(fem_mag_u, x=xnp, y=ynp, str="fem" + "_mag_u_" + str(args.mu))
    sp.plot(fem_mag_z, x=xnp, y=ynp, str="fem" + "_mag_z_" + str(args.mu))

# Se errors=1 allora plotto tutti gli errori che servono
if args.errors:
    fem_vx = torch.tensor(fem_vxnp, dtype=torch.float).view(-1, 1)
    fem_vy = torch.tensor(fem_vynp, dtype=torch.float).view(-1, 1)
    fem_ux = torch.tensor(fem_uxnp, dtype=torch.float).view(-1, 1)
    fem_uy = torch.tensor(fem_uynp, dtype=torch.float).view(-1, 1)
    fem_p = torch.tensor(fem_pnp, dtype=torch.float).view(-1, 1)
    fem_r = torch.tensor(fem_rnp, dtype=torch.float).view(-1, 1)
    fem_zx = torch.tensor(fem_zxnp, dtype=torch.float).view(-1, 1)
    fem_zy = torch.tensor(fem_zynp, dtype=torch.float).view(-1, 1)

    X = torch.tensor(xnp, dtype=torch.float).view(-1, 1)
    Y = torch.tensor(ynp, dtype=torch.float).view(-1, 1)

    # Qui faccio dei magheggi solo per avere coerenza nel momento in cui valuto la rete neurale
    x_labelT = LabelTensor(X, ['x'])
    y_labelT = LabelTensor(Y, ['y'])
    mu_labelT = LabelTensor(torch.full((n,), args.mu, dtype=torch.float).view(-1, 1), ['mu'])
    input = x_labelT.append(y_labelT).append(mu_labelT)

    # Qua la rete sputa fuori i suoi risultati
    output = pinn.forward(input.cuda())

    # Nel caso stokes ho bisogno di ridefinire le variabili vettoriali tramite il loro modulo, perche e piu facile da trattare
    v_magnitude = np.sqrt(
        output.extract(['vx']).cpu().detach().numpy() ** 2 + output.extract(['vy']).cpu().detach().numpy() ** 2)
    u_magnitude = np.sqrt(
        output.extract(['ux']).cpu().detach().numpy() ** 2 + output.extract(['uy']).cpu().detach().numpy() ** 2)
    z_magnitude = np.sqrt(
        output.extract(['zx']).cpu().detach().numpy() ** 2 + output.extract(['zy']).cpu().detach().numpy() ** 2)

    # Li stampo subito qui
    sp.plot(v_magnitude, x=xnp, y=ynp, str="mag_v")
    sp.plot(u_magnitude, x=xnp, y=ynp, str="mag_u")
    sp.plot(z_magnitude, x=xnp, y=ynp, str="mag_z")

    # Qui bisogna definirli come tensori perché altrimenti non riesce a trasferire su cpu
    # Errore assoluto
    errore_vx = abs(
        LabelTensor((output.extract(['vx']) - fem_vx.cuda()).reshape(n, ), ['errore_vx']).cpu().detach().numpy())
    errore_vy = abs(
        LabelTensor((output.extract(['vy']) - fem_vy.cuda()).reshape(n, ), ['errore_vy']).cpu().detach().numpy())
    errore_ux = abs(
        LabelTensor((output.extract(['ux']) - fem_ux.cuda()).reshape(n, ), ['errore_ux']).cpu().detach().numpy())
    errore_uy = abs(
        LabelTensor((output.extract(['uy']) - fem_uy.cuda()).reshape(n, ), ['errore_uy']).cpu().detach().numpy())
    errore_p = abs(
        LabelTensor((output.extract(['p']) - fem_p.cuda()).reshape(n, ), ['errore_p']).cpu().detach().numpy())
    errore_r = abs(
        LabelTensor((output.extract(['r']) - fem_r.cuda()).reshape(n, ), ['errore_r']).cpu().detach().numpy())
    errore_zx = abs(
        LabelTensor((output.extract(['zx']) - fem_zx.cuda()).reshape(n, ), ['errore_zx']).cpu().detach().numpy())
    errore_zy = abs(
        LabelTensor((output.extract(['zy']) - fem_zy.cuda()).reshape(n, ), ['errore_zy']).cpu().detach().numpy())

    # Erroe relativo
    err_rel_vx = errore_vx / np.linalg.norm(fem_vx)
    err_rel_vy = errore_vy / np.linalg.norm(fem_vy)
    err_rel_ux = errore_ux / np.linalg.norm(fem_ux)
    err_rel_uy = errore_uy / np.linalg.norm(fem_uy)
    err_rel_p = errore_p / np.linalg.norm(fem_p)
    err_rel_r = errore_r / np.linalg.norm(fem_r)
    err_rel_zx = errore_zx / np.linalg.norm(fem_zx)
    err_rel_zy = errore_zy / np.linalg.norm(fem_zy)

    #####################
    # Grafici degli errori relativi e assoluti calcolati rispetto alla soluzione fem
    sp = Scatter_plot()
    if (args.physic_sampling == "grid" or args.physic_sampling == 'chebyshev'):
        sp.plots_grid(err_rel_vx, x=xnp, y=ynp, str="errore_rel" + suf_vx, sample_points=sample_points)
        sp.plots_grid(err_rel_vy, x=xnp, y=ynp, str="errore_rel" + suf_vy, sample_points=sample_points)
        sp.plots_grid(err_rel_ux, x=xnp, y=ynp, str="errore_rel" + suf_ux, sample_points=sample_points)
        sp.plots_grid(err_rel_uy, x=xnp, y=ynp, str="errore_rel" + suf_uy, sample_points=sample_points)
        sp.plots_grid(err_rel_p, x=xnp, y=ynp, str="errore_rel" + suf_p, sample_points=sample_points)
        sp.plots_grid(err_rel_r, x=xnp, y=ynp, str="errore_rel" + suf_r, sample_points=sample_points)
        sp.plots_grid(err_rel_zx, x=xnp, y=ynp, str="errore_rel" + suf_zx, sample_points=sample_points)
        sp.plots_grid(err_rel_zy, x=xnp, y=ynp, str="errore_rel" + suf_zy, sample_points=sample_points)

        sp.plots_grid(errore_vx, x=xnp, y=ynp, str="errore" + suf_vx, sample_points=sample_points)
        sp.plots_grid(errore_vy, x=xnp, y=ynp, str="errore" + suf_vy, sample_points=sample_points)
        sp.plots_grid(errore_ux, x=xnp, y=ynp, str="errore" + suf_ux, sample_points=sample_points)
        sp.plots_grid(errore_uy, x=xnp, y=ynp, str="errore" + suf_uy, sample_points=sample_points)
        sp.plots_grid(errore_p, x=xnp, y=ynp, str="errore" + suf_p, sample_points=sample_points)
        sp.plots_grid(errore_r, x=xnp, y=ynp, str="errore" + suf_r, sample_points=sample_points)
        sp.plots_grid(errore_zx, x=xnp, y=ynp, str="errore" + suf_zx, sample_points=sample_points)
        sp.plots_grid(errore_zy, x=xnp, y=ynp, str="errore" + suf_zy, sample_points=sample_points)
    else:
        sp.plots(err_rel_vx, x=xnp, y=ynp, str="errore_rel" + suf_vx, sample_points=sample_points)
        sp.plots(err_rel_vy, x=xnp, y=ynp, str="errore_rel" + suf_vy, sample_points=sample_points)
        sp.plots(err_rel_ux, x=xnp, y=ynp, str="errore_rel" + suf_ux, sample_points=sample_points)
        sp.plots(err_rel_uy, x=xnp, y=ynp, str="errore_rel" + suf_uy, sample_points=sample_points)
        sp.plots(err_rel_p, x=xnp, y=ynp, str="errore_rel" + suf_p, sample_points=sample_points)
        sp.plots(err_rel_r, x=xnp, y=ynp, str="errore_rel" + suf_r, sample_points=sample_points)
        sp.plots(err_rel_zx, x=xnp, y=ynp, str="errore_rel" + suf_zx, sample_points=sample_points)
        sp.plots(err_rel_zy, x=xnp, y=ynp, str="errore_rel" + suf_zy, sample_points=sample_points)

        sp.plots(errore_vx, x=xnp, y=ynp, str="errore" + suf_vx, sample_points=sample_points)
        sp.plots(errore_vy, x=xnp, y=ynp, str="errore" + suf_vy, sample_points=sample_points)
        sp.plots(errore_ux, x=xnp, y=ynp, str="errore" + suf_ux, sample_points=sample_points)
        sp.plots(errore_uy, x=xnp, y=ynp, str="errore" + suf_uy, sample_points=sample_points)
        sp.plots(errore_p, x=xnp, y=ynp, str="errore" + suf_p, sample_points=sample_points)
        sp.plots(errore_r, x=xnp, y=ynp, str="errore" + suf_r, sample_points=sample_points)
        sp.plots(errore_zx, x=xnp, y=ynp, str="errore" + suf_zx, sample_points=sample_points)
        sp.plots(errore_zy, x=xnp, y=ynp, str="errore" + suf_zy, sample_points=sample_points)

    sp.plot(output.extract(['vx']).cpu().detach().numpy(), x=xnp, y=ynp, str='vx_' + str(args.mu))
    sp.plot(output.extract(['vy']).cpu().detach().numpy(), x=xnp, y=ynp, str='vy_' + str(args.mu))
    sp.plot(output.extract(['p']).cpu().detach().numpy(), x=xnp, y=ynp, str='p_' + str(args.mu))
    sp.plot(output.extract(['ux']).cpu().detach().numpy(), x=xnp, y=ynp, str='ux_' + str(args.mu))
    sp.plot(output.extract(['uy']).cpu().detach().numpy(), x=xnp, y=ynp, str='uy_' + str(args.mu))
    sp.plot(output.extract(['r']).cpu().detach().numpy(), x=xnp, y=ynp, str='r_' + str(args.mu))
    sp.plot(output.extract(['zx']).cpu().detach().numpy(), x=xnp, y=ynp, str='zx_' + str(args.mu))
    sp.plot(output.extract(['zy']).cpu().detach().numpy(), x=xnp, y=ynp, str='zy_' + str(args.mu))

    #####################################################################
    # Calcolo errore in norma l2 (norma della differenza)#################
    #####################################################################

    norma_errore_vx = np.linalg.norm(errore_vx) / np.linalg.norm(fem_vxnp)
    norma_errore_vy = np.linalg.norm(errore_vy) / np.linalg.norm(fem_vynp)
    norma_errore_ux = np.linalg.norm(errore_ux) / np.linalg.norm(fem_uxnp)
    norma_errore_uy = np.linalg.norm(errore_uy) / np.linalg.norm(fem_uynp)
    norma_errore_p = np.linalg.norm(errore_p) / np.linalg.norm(fem_pnp)
    norma_errore_r = np.linalg.norm(errore_r) / np.linalg.norm(fem_rnp)
    norma_errore_zx = np.linalg.norm(errore_zx) / np.linalg.norm(fem_zxnp)
    norma_errore_zy = np.linalg.norm(errore_zy) / np.linalg.norm(fem_zynp)

    abs_err_vx = np.linalg.norm(errore_vx)
    abs_err_vy = np.linalg.norm(errore_vy)
    abs_err_ux = np.linalg.norm(errore_ux)
    abs_err_uy = np.linalg.norm(errore_uy)
    abs_err_p = np.linalg.norm(errore_p)
    abs_err_r = np.linalg.norm(errore_r)
    abs_err_zx = np.linalg.norm(errore_zx)
    abs_err_zy = np.linalg.norm(errore_zy)

    with open('l2_errors_' + str(args.mu) + '.txt', 'w') as file:
        file.write(f"errore relativo_vx = {norma_errore_vx}\n")
        file.write(f"errore relativo_vy = {norma_errore_vy}\n")
        file.write(f"errore relativo_ux = {norma_errore_ux}\n")
        file.write(f"errore relativo_uy = {norma_errore_uy}\n")
        file.write(f"errore relativo_p = {norma_errore_p}\n")
        file.write(f"errore relativo_r = {norma_errore_r}\n")
        file.write(f"errore relativo_zx = {norma_errore_zx}\n")
        file.write(f"errore relativo_zy = {norma_errore_zy}\n")

        file.write(f"errore assoluto_vx = {abs_err_vx}\n")
        file.write(f"errore assoluto_vy = {abs_err_vy}\n")
        file.write(f"errore assoluto_ux = {abs_err_ux}\n")
        file.write(f"errore assoluto_uy = {abs_err_uy}\n")
        file.write(f"errore assoluto_p = {abs_err_p}\n")
        file.write(f"errore assoluto_r = {abs_err_r}\n")
        file.write(f"errore assoluto_zx = {abs_err_zx}\n")
        file.write(f"errore assoluto_zy = {abs_err_zy}\n")


