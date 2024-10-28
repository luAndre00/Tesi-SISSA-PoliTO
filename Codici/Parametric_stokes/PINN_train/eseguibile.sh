#Faccio vari training

#Baseline
nohup python3 main_PINN.py --npname "baseline" --neurons 100 --physic_sampling "latin" --parametric_sampling "latin" --func "softplus"
#Strong
nohup python3 main_PINN.py --npname "strong" --neurons 100 --physic_sampling "latin" --parametric_sampling "latin" --func "softplus" --strong 1  
#Strong-Grid
nohup python3 main_PINN.py --npname "strong_grid" --neurons 100 --physic_sampling "grid" --parametric_sampling "latin" --func "softplus" --strong 1  
#Grid-Sampling
nohup python3 main_PINN.py --npname "grid" --neurons 100 --physic_sampling "grid" --parametric_sampling "latin" --func "softplus"  
#Chebyshev Sampling
nohup python3 main_PINN.py --npname "chebyshev" --neurons 100 --physic_sampling "chebyshev" --parametric_sampling "latin" --func "softplus"  
#Fourier-Embedding
nohup python3 main_PINN.py --npname "fourier" --neurons 100 --physic_sampling "latin" --parametric_sampling "latin" --func "softplus" --fourier 1  
#Neurons
nohup python3 main_PINN.py --npname "200_neurons" --neurons 200 --physic_sampling "latin" --parametric_sampling "latin" --func "softplus"  
#Fourier-tanh
nohup python3 main_PINN.py --npname "fourier_tanh" --neurons 100 --physic_sampling "latin" --parametric_sampling "latin" --func "tanh" --fourier 1  


