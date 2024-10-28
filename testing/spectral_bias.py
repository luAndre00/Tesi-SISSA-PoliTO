
import torch
import argparse
import numpy as np
from pina import LabelTensor
from pina.solvers import PINN
from pina.model import MultiFeedForward
from pina.plotter import Plotter
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.callbacks import MetricTracker

from pina import Condition
from pina.geometry import CartesianDomain
from pina.equation import Equation, FixedValue
from pina.problem import SpatialProblem
from pina.operators import laplacian
from pina.model.layers import FourierFeatureEmbedding
import matplotlib.pyplot as plt

class MultiscaleFourierNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embedding1 = FourierFeatureEmbedding(input_dimension=1, output_dimension=100, sigma=1)
        self.embedding2 = FourierFeatureEmbedding(input_dimension=1, output_dimension=100, sigma=10)
        self.layers = FeedForward(*args, **kwargs)
        self.final_layer = torch.nn.Linear(2*100, 1)

    def forward(self, x):
        e1 = self.layers(self.embedding1(x))
        e2 = self.layers(self.embedding2(x))
        out = self.final_layer(torch.cat([e1, e2], dim=-1))
        out.labels = ['u']
        return out
    

class Poisson(SpatialProblem):    
    xmin = -np.pi
    xmax = np.pi
    xrange = [xmin, xmax]
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': xrange})
    
    def residual(input_, output_):
        lap = laplacian(output_, input_, components = ['u'], d = ['x'])
        forzante = torch.sin(input_.extract(['x'])) + 0.1 * (alpha ** 2) * torch.sin(alpha*input_.extract(['x']))
        return  lap + forzante

    def truth_solution(self, input_):
        return (torch.sin(input_.extract(['x'])) + 0.1 * torch.sin(alpha*input_.extract(['x'])))
    
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
    parser.add_argument("--epochs", help="epochs of training", type=int, default=1200)
    parser.add_argument("--alpha", help="frequency of the solution", type=int, default = 1)
    parser.add_argument("--lr", help="learning rate", type=float, default = 0.0005)
    parser.add_argument("--sigma", help="variance for the Fourier embedding", type=float, default = 1)
    parser.add_argument("--fourier", help="Whether to use fourier embedding", type=int, default = 1)
    args = parser.parse_args()

    alpha = torch.tensor(args.alpha)
    
    torch.manual_seed(seed); problem.discretise_domain(n=1, mode='random', variables=['x'], locations=['gamma1'])
    torch.manual_seed(seed); problem.discretise_domain(n=1, mode='random', variables=['x'], locations=['gamma2'])
    torch.manual_seed(seed); problem.discretise_domain(n=200, mode='grid', variables=['x'], locations=['D'])

    class sin(torch.nn.Module):
        def __init__(self):
            super(sin, self).__init__()
            return
            
        def forward(self, x):
            return torch.sin(x)    
    
    if args.fourier == 1:
        model = MultiscaleFourierNet(layers = [100], func = torch.nn.Tanh, output_dimensions = 100, input_dimensions = 100, )
    else:
        model = FeedForward(layers = [100, 100], func = torch.nn.Softplus, output_dimensions = 1, input_dimensions = 1)

  #  checkp = '/scratch/atataran/Tesi-SISSA-PoliTO/testing/test.ckpt'
   # pinn = PINN.load_from_checkpoint(checkpoint_path=checkp, problem=problem, model=model)


    pinn = PINN(problem=problem, model=model, optimizer_kwargs = {'lr': args.lr})
    track = MetricTracker()
    trainer = Trainer(solver=pinn, accelerator='gpu', max_epochs=args.epochs, callbacks = [track])
    trainer.train()

    plotter = Plotter()
    plotter.plot(solver = pinn, filename = "_alpha_" + str(alpha.item()) + ".png")
    plotter.plot_loss(trainer = trainer, filename = '_loss' + str(alpha.item()) + '.png', metrics = ['D_loss', 'gamma1_loss', 'gamma2_loss'], logx = True, logy = True)


    #Prendo i valori della loss manualmente
    res_loss = track.metrics['D_loss'].cpu().numpy().reshape((args.epochs),1)





#####################################################################################################################

    #QUESTO E UN PICCOLO ESEMPIO DI COME POSSO CALCOLARE LE DERIVATE E LA LOSS PER LE PINN, PUO RISULTARE MOLTO UTILE
    #Valuto il residuo della soluzione analitica: verificato! Viene 1e-11
    #Valutare la differenza dei laplaciani eventualmente con plot Verificato! I laplaciani sono UGUALI
    #Plot del residuo analitico con anche il residuo della pinn
    #Provare a usare una loss normalizzata per vedere se è più alta quale loss provo?

    input = problem.input_pts['D']['x'] #punti di training in torch
    input = LabelTensor(input, ['x'])
    input.requires_grad_(True)
    output = pinn.forward(input)
    lap = torch.zeros_like(output)
    
    for i in range(output.size(0)):
        inut = torch.ones_like(output[i])
        grad_first = torch.autograd.grad(outputs=output[i], inputs=input, grad_outputs=inut, create_graph=True)[0]
        grad_second = torch.autograd.grad(grad_first[i], input, grad_outputs=torch.ones_like(grad_first[i]), create_graph=True)[0]
        lap[i]=grad_second[i]

    true_lap = (-(torch.sin(input) + 0.1 * (alpha ** 2) * torch.sin(alpha*input))).detach().numpy()
    lap = lap.detach().numpy()
    num_input = input.detach().numpy()
    
    #Qui invece calcolo il residuo
    forzante = torch.sin(input) + 0.1 * (alpha ** 2) * torch.sin(alpha*input)
    res_pinn = abs(forzante.detach().numpy() + lap)
    true_res = abs(true_lap + forzante.detach().numpy())
    laplacian_res = abs(true_lap-lap)/np.linalg.norm(true_lap)
    print("")
    print("L'errore relativo sui laplaciano vale", laplacian_res)
    print("")
    
    plt.plot(num_input, res_pinn, label='true', color='blue', linewidth = 2) 
    plt.plot(num_input, true_res, label='true', color='red', linewidth = 2) 
    plt.savefig('grafico_residui')
    plt.close()

    plt.plot(num_input, true_lap, label='true', color='blue', linewidth = 2) 
    plt.plot(num_input, lap, label='app', color='red')   
    plt.savefig('laplaciani.png')
    plt.close()
        

    

#####################################################################################################################













########################################################################################################################

    #punti_training = problem.input_pts['D']['x'] #punti di training in torch
    #input = LabelTensor(punti_training, ['x'])
    #input.requires_grad_(True)
    #output = pinn.forward(input) #QUESTO DIVENTA ESATTAMENTE LA FORZANTE, la loss deve essere 0

    #lap = torch.zeros_like(output)
    #for i in range(output.size(0)):
    #    inut = torch.ones_like(output[i])
    #    grad_first = torch.autograd.grad(outputs=output[i], inputs=input, grad_outputs=inut, create_graph=True)[0]
    #    grad_second = torch.autograd.grad(grad_first[i], input, grad_outputs=torch.ones_like(grad_first[i]), create_graph=True)[0]
    #    lap[i]=grad_second[i]

    #forzante = torch.sin(input) + 0.1 * (alpha ** 2) * torch.sin(alpha*input)
    #res = (forzante + lap).detach().numpy()
    #loss = np.mean(res*res)
    #print(loss)
    #print(res_loss[args.epochs-1])


    





